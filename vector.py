from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path



def process_document_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    documents = [
    Document(page_content=text, metadata={"source": str(file_path), "chunk_id": i})
        for i, text in enumerate(chunks)
    ]

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore
    

vectorstore = process_document_file(Path("data") / "test.txt")




results = vectorstore.similarity_search_with_score('Tell me about beverage trade?', k=1)

for doc, score in results:
    print(f"Score: {score:.3f}")
    print(f"Text: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")