import os
import json
import time
import textwrap
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from google import genai
from google.genai import types

from retrieve import retrieve_documents 
from dataset import load_dataset

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("model")
rate_limit = int(os.getenv("rate_limit"))
waiting_time = int(os.getenv("waiting_time"))
output_file = "result/rag.json"

# ------------------------------------------
system_prompt = """You are an expert Q&A assistant. Your task is to answer the user's question strictly based on the provided Context.

Rules:
1. NO HALLUCINATION: You must base your answer ONLY on the information found in the Context. Do not use your pre-trained external knowledge.
2. MISSING INFO: If the Context does not contain the answer to the question, you must explicitly output: "I cannot answer this based on the provided context."
3. BE CONCISE: Provide a direct, clear, and concise answer without unnecessary filler words.
4. Once you have enough information, output your final answer to the user."""
# ------------------------------------------

def call_llm(model_name, client, system_prompt, user_prompt):
    """call Gemini for QA Task"""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.0 
    )
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=config
        )
        return response.text
    except Exception as e:
        return f"LLM Error: {str(e)}"

def main():
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

    # load dataset
    _, qa_pairs = load_dataset()

    results_log = []
    for i, qa_pair in enumerate(qa_pairs):
        query = qa_pair['query']
        ground_truth = qa_pair['answer']

        retrieved_docs = retrieve_documents(query, collection, top_k=5)
        user_prompt = textwrap.dedent(f"""
            Context information is below:
            ---------------------
            {retrieved_docs}
            ---------------------

            Question: {query}

            Answer:
        """).strip()

        response = call_llm(model, client, system_prompt, user_prompt)

        print(f"Test [{i+1}/{len(qa_pairs)}]")
        print(f"🤔 Question: {query}")
        print(f"🎯 Ground Truth: {ground_truth}")
        print(f"🤖 LLM Answer:   {response}")
        print("-" * 60)

        results_log.append({
            "id": i + 1,
            "query": query,
            "ground_truth": ground_truth,
            "llm_answer": response,
        })

        if (i + 1) % rate_limit == 0 and (i + 1) != len(qa_pairs):
            print(f"Hitted Rate Limit, waiting for {waiting_time} second.")
            time.sleep(waiting_time)

    # save as json
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)
        
    print(f"Run Finished, result saved to: {output_file}")

if __name__ == "__main__":
    main()