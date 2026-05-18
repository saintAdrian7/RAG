import os
import yaml
from typing import List
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_documents() -> List[Document]:
    """
    Load documents for demonstration.

    Returns:
        List of LangChain Document objects loaded from the ./data directory.
    """
    results: List[Document] = []
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    for file in os.listdir(DATA_DIR):
        if file.endswith('.txt'):
            file_path = os.path.join(DATA_DIR, file)
            try:
                loader = TextLoader(file_path)
                loader_docs = loader.load()
                results.extend(loader_docs)
                print(f"Successfully added {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    print(f"\nTotal documents loaded: {len(results)}")
    return results

def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
