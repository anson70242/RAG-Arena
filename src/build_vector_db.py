import json
import chromadb
from dataset import load_dataset
from chromadb.utils import embedding_functions

def build_vector_db():
    # initialize the Chroma client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # load the embedding function
    print("Loading embedding function...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # create a collection for the documents
    collection = chroma_client.get_or_create_collection(
        name="multihop_demo",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    # load the dataset
    corpus, mini_qa = load_dataset()
    documents = [doc["body"] for doc in corpus]
    metadatas = [{"title": doc["title"]} for doc in corpus] 
    ids = [str(i) for i in range(len(documents))]

    print(f"Adding {len(documents)} documents to the Chroma collection...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Finished adding documents to the Chroma collection.")

if __name__ == "__main__":
    build_vector_db()