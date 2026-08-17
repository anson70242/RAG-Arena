import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

def retrieve_documents(query, collection, top_k=5):
    print(f"Retrieving top {top_k} documents for the query: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    for i, (metadata, score) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        print(f"Rank {i + 1}:")
        print(f"Title: {metadata['title']}")
        print(f"Score: {score}")
        print("-" * 50)

    return results

def retrieve_with_rerank(query, collection, reranker, initial_top_k=15, final_top_k=5):
    """Add reranker"""
    print(f"Retrieving top {initial_top_k} documents for the query: '{query}'")
    results = collection.query(
        query_texts=[query],
        n_results=initial_top_k
    )

    initial_docs = results['documents'][0]
    initial_metas = results['metadatas'][0]

    print("Perfoming Reranking")
    sentence_pairs = [[query, doc] for doc in initial_docs]
    rerank_scores = reranker.predict(sentence_pairs)

    paired_results = list(zip(initial_docs, initial_metas, rerank_scores))
    paired_results.sort(key=lambda x: x[2], reverse=True)

    final_results = paired_results[:final_top_k]

    for i, (doc, meta, score) in enumerate(final_results):
        print(f"Rank {i + 1}:")
        print(f"Title: {meta.get('title', 'Unknown')}")
        print(f"Rerank Score: {score:.4f}")
        print("-" * 50)
        
    return [doc for doc, meta, score in final_results]


if __name__ == "__main__":
    print("Initializing Chroma client...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    print("Loading embedding function...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

    collection = chroma_client.get_collection(
        name="multihop_demo",
        embedding_function=embedding_function
    )

    sample_query = "Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?"
    retrieve_documents(sample_query, collection, top_k=5)

    print("Loading Reranker model...")
    reranker = CrossEncoder('BAAI/bge-reranker-base')
    retrieve_with_rerank(sample_query, collection, reranker, initial_top_k=15, final_top_k=5)