# Development of a PDF-Based Question-Answering Chatbot Using LangChain

## AIM

To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

## PROBLEM STATEMENT

Develop a chatbot that can understand the contents of a PDF document and answer user questions based on the information contained within the document. The system should retrieve relevant sections of the PDF and use a Large Language Model (LLM) to generate meaningful answers.

## TECHNOLOGIES USED

- Python
- LangChain
- OpenAI Embeddings
- OpenAI GPT-4o-mini
- FAISS
- PyPDF
- Jupyter Notebook

## SYSTEM ARCHITECTURE

```text
PDF Document
     ↓
PDF Text Extraction
     ↓
Text Splitting
     ↓
Document Chunks
     ↓
OpenAI Embeddings
     ↓
FAISS Vector Store
     ↓
Retriever
     ↓
Relevant Document Chunks
     ↓
GPT-4o-mini
     ↓
Generated Answer
```

## DESIGN STEPS

### STEP 1: Load and Extract PDF Content

The PDF document is loaded using `PyPDFLoader`. The contents of the PDF are extracted and stored as document objects.

### STEP 2: Split the Document into Chunks

The extracted text is divided into smaller overlapping chunks using `RecursiveCharacterTextSplitter`.

A chunk size of 1000 characters and an overlap of 200 characters are used.

### STEP 3: Create Embeddings and Vector Store

Each document chunk is converted into a numerical vector using OpenAI Embeddings. These vectors are stored in a FAISS vector store for efficient similarity-based retrieval.

### STEP 4: Create the Retriever

A retriever is created from the FAISS vector store. When the user asks a question, the retriever searches for the most relevant document chunks.

### STEP 5: Generate the Answer

The retrieved chunks are passed as context to GPT-4o-mini. The model generates an answer based only on the retrieved information.

## PROGRAM:
```
import os
import sys
import site

# Make sure user-installed packages are visible
site.addsitedir(site.getusersitepackages())

# LangChain imports compatible with LangChain 0.0.312
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

print("All imports successful!")
```
```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("GenoMatch_Project_Report (1).pdf")
documents = loader.load()

print("Number of pages:", len(documents))
```

```
print("Page 1 text length:", len(documents[0].page_content))
print("Page 1 text:", repr(documents[0].page_content[:500]))
```
```
print("Text length:", len(documents[0].page_content))
print(documents[0].page_content[:1000])
```

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0].page_content)
```
```
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

print("Embedding model initialized!")
```
```
from langchain.vectorstores import FAISS

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS vector store created!")
```
```
retriever = vector_store.as_retriever()

print("Retriever created!")
```
```
question = "What is the primary goal of GenoMatch?"

relevant_docs = retriever.get_relevant_documents(question)

print("Number of relevant chunks:", len(relevant_docs))

for i, doc in enumerate(relevant_docs):
    print(f"\n--- Relevant Chunk {i+1} ---")
    print(doc.page_content)
```

```
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

print("LLM initialized!")
```

```
context = "\n\n".join(
    doc.page_content for doc in relevant_docs
)

prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.predict(prompt)

print(response)
```
## OUTPUT:
<img width="1080" height="397" alt="ex03" src="https://github.com/user-attachments/assets/5c89515d-b467-4eef-9365-d3680b937d0e" />


The chatbot successfully retrieves relevant information from the PDF and generates an appropriate answer to the user's query.

## RESULT

The PDF-based question-answering chatbot was successfully designed and implemented using LangChain. The system extracts text from the PDF, divides it into chunks, converts the chunks into embeddings, stores them in FAISS, retrieves relevant information for a given query, and uses GPT-4o-mini to generate the final answer.

The system was successfully tested with queries derived from the PDF document and produced relevant responses.

