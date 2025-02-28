import os
import pandas as pd
import chardet
import openpyxl
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


# ---------- Helper Function to Read TXT Files with Auto Encoding ----------
def read_file_with_auto_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            text = f.read()
            # Remove BOM if present
            if text.startswith('\ufeff'):
                text = text[1:]
            return text.strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


# ---------- Step 1: Process Files and Build Aggregated Data ----------
folder_path = "/Users/jiayaochen/Desktop/TextAnnotation/DowngradeFullForLabel"

data = []  # List to store data rows
file_report = []  # List to store per-file integrity reports

for filename in os.listdir(folder_path):
    if filename.startswith('~$'):
        continue  # Skip temporary files

    file_path = os.path.join(folder_path, filename)
    report = {"filename": filename}

    if filename.endswith('.txt'):
        report["type"] = "txt"
        text = read_file_with_auto_encoding(file_path)
        report["length"] = len(text)
        report["is_empty"] = (len(text.strip()) == 0)
        data.append({
            "speech_name": filename,  # Using the file name as the unique identifier
            "text": text,
            "source": "txt"
        })

    elif filename.endswith('.xlsx'):
        report["type"] = "excel"
        try:
            df_excel = pd.read_excel(file_path, engine='openpyxl')
            num_rows = len(df_excel)
            report["num_rows"] = num_rows
            if "content" in df_excel.columns:
                report["missing_content_count"] = int(df_excel["content"].isnull().sum())
                report["empty_content_count"] = int(df_excel["content"].apply(lambda x: len(str(x).strip()) == 0).sum())
            else:
                report["content_column_missing"] = True

            # Process each row in the Excel file, assigning a unique id for each row
            for idx, row in df_excel.iterrows():
                text_content = str(row.get("content", "")).strip()
                unique_id = f"{filename}_row{idx}"
                data.append({
                    "speech_name": unique_id,
                    "text": text_content,
                    "stance": row.get("Stance", "N/A"),
                    "satire": row.get("Satire presence", "N/A"),
                    "ambiguity": row.get("Ambiguity", "N/A"),
                    "source": "excel"
                })
        except Exception as e:
            report["error"] = str(e)

    else:
        report["type"] = "other"

    file_report.append(report)

# ---------- Step 2: File Integrity Report ----------
print("=== File Integrity Report ===")
for rep in file_report:
    print(rep)

# ---------- Step 3: Aggregated DataFrame Integrity Check ----------
df = pd.DataFrame(data)
print("\nTotal rows aggregated from all files:", len(df))

# Check unique identifiers
unique_speeches = df['speech_name'].nunique()
print("Unique speech names:", unique_speeches)

# Group by 'speech_name' to see how many rows per unique identifier (should be 1 per row if unique)
speech_counts = df.groupby('speech_name').size().reset_index(name="row_count")
print("\nRows per speech_name:")
print(speech_counts)

# Check for empty text entries
df["text"] = df["text"].astype(str).fillna("")
empty_text_count = (df["text"].str.strip() == "").sum()
print(f"\nNumber of rows with empty text: {empty_text_count}")

# ---------- Step 4: Optional Text Splitting Using LangChain ----------
# Initialize the tokenizer for GPT-4o (adjust model if needed)
encoding = tiktoken.encoding_for_model("gpt-4o")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    length_function=lambda text: len(encoding.encode(text)),
    separators=["\n\n", "\n", " ", ""]
)

# Create a new column for text chunks
df["chunks"] = df["text"].apply(lambda x: text_splitter.split_text(x) if isinstance(x, str) and x.strip() != "" else [])
df["chunk_count"] = df["chunks"].apply(len)

print("\nChunk count statistics:")
print(df["chunk_count"].describe())

print("\nSample entries with chunks:")
print(df[["speech_name", "chunk_count", "chunks"]].head(10).to_string(index=False))

import time
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import LLMChain
from langchain_openai import ChatOpenAI

# ------------------ Part 5: Labeling with GPT-4o ------------------

# Define system and human prompts.
system_prompt = (
    "你是一名擅长分析政治文本的专家，你的任务是为以下文本进行标注，判断其是否包含讽刺（satire）元素。"
)

human_prompt = (
    "你的目标是评估以下政治文本中讽刺的存在情况:\n"
    "{text}\n\n"
    "讽刺 (Satire) 必须包含以下至少一种元素:\n"
    "1. **文化参照**: 文化参照借鉴了共通的社会经验--例如国家叙事、文化载体和共有的假设--来突出这些观点和现实之间的冲突。文化参照通常采用以下形式：\n"
    "   - 反转常见叙事，重新措辞/定义  \n"
    "   - 诉诸常识/逻辑，借鉴虚构角色/时间/地点  \n"
    "   - **运用同音异义 (homophones) 或细微语言变形来讽刺某种叙事或群体**  \n"
    "   - 如果文本涉及特定背景（如网络热门话题、政策讨论、或国家科技发展），请分析其是否以讽刺方式回应这些背景。\n\n"
    "2. **句式修改**: 通过调整单词、短语或句子结构来达成讽刺效果。包括：\n"
    "   - 并置对立或不相关的观点  \n"
    "   - 文字游戏/双重语义  \n"
    "   - 以提问代替直接陈述  \n\n"
    "3. **非字面意义**: 语言表面表达某种观点，实则暗示相反含义，或故意夸张的解决方案，以揭示问题中的矛盾和非理性。\n\n"
    "4. **讽刺态度 (Sarcaticism)**: 通过挖苦、夸张对比、戏剧化情境表达愤世嫉俗或批判，通常依赖语气、文本前后关系、对比等。\n\n"
    "---\n"
    "#### **示例 1:**\n"
    "- **文本:** “且听笼吟。”\n"
    "- **分析:** 该句为“且听龙吟”的**谐音变体**，用“笼”代替“龙”，形成**语言变形**，暗示对盲目民族主义的讽刺。由于它模仿了民族主义话语，并将其转换为相反意义，因此是讽刺。\n"
    "- **结论:** **是: 讽刺通过【文化参照, 非字面意义】表现。**\n\n"
    "### **重要:**\n"
    "- **如果文本出现 2 个或以上讽刺特征，请标记为 \"是\" (Yes, Satire)。**\n"
    "- **仅当文本完全没有讽刺手法时，才选择 \"否\" (No, Satire)。**\n"
    "- **如果文本修改了常见国家叙事 (如科技、经济崛起) 并使用了双关或谐音，请特别留意可能的讽刺含义。**\n"
    "- **你不需要在回答中给出分析和结论**"
)

# Create the prompt template.
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", human_prompt),
])

# Initialize the GPT-4o model.
load_dotenv(find_dotenv())
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.3)

# Create an LLMChain with an output parser (to ensure we extract a simple response).
chain = LLMChain(prompt=prompt_template, llm=llm, output_parser=StrOutputParser())

# ------------------ Process Each Row for Labeling ------------------

# Prepare a list to store labeling results.
results = []

# Iterate over each row in the DataFrame.
# (Assumes that df has already been built with a "chunks" column from earlier steps.)
for index, row in df.iterrows():
    chunk_labels = []
    chunk_justifications = []

    # Process each text chunk using GPT-4o.
    for chunk_idx, chunk in enumerate(row["chunks"]):
        try:
            # Get response from GPT-4o.
            response = chain.run({"text": chunk})
            # Expected response is either "是" or "否", optionally with a colon and explanation.
            if ":" in response:
                label, justification = response.split(":", 1)
                label = label.strip()
                justification = justification.strip()
            else:
                label = response.strip()
                justification = ""
            chunk_labels.append(label)
            chunk_justifications.append(justification)

            # Delay between API calls to help avoid rate limits.
            time.sleep(1)
        except Exception as e:
            print(f"Error processing row {index}, chunk {chunk_idx}: {e}")
            continue

    # Determine overall satire label: if any chunk is labeled "是", mark the entire speech as "是".
    final_label = "是" if "是" in chunk_labels else "否"

    results.append({
        "speech_name": row["speech_name"],
        "text": row["text"],
        "satire_presence": final_label,
        "num_chunks_processed": len(chunk_labels),
        "chunk_labels": chunk_labels,
        "chunk_justifications": chunk_justifications
    })

# Convert results into a DataFrame.
df_results = pd.DataFrame(results)

# Optionally, save the labeling results to a CSV file.
df_results.to_csv("Test1temp03.csv", index=False)
print("✅ Satire detection completed and saved to 'Test1Temp03.csv'")
