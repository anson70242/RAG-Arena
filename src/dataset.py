import os
import json
from datasets import load_dataset as hf_load_dataset

def create_matched_data():
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Load the QA pairs
    mini_qa = hf_load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG", split="train[:20]")

    # Filter the corpus to only include documents that are referenced in the mini QA dataset
    required_titles = set()
    for qa in mini_qa:
        for evidence in qa.get("evidence_list", []):
            required_titles.add(evidence["title"])

    print(f"This 10 QA pairs relate to {len(required_titles)} documents.")

    full_corpus = hf_load_dataset("yixuantt/MultiHopRAG", "corpus", split="train")
    matched_docs = []
    distraction_docs = []

    for doc in full_corpus:
        if doc["title"] in required_titles:
            matched_docs.append(doc)
        else:
            # Add some noise to test the retrieval capabilities of the system
            distraction_docs.append(doc)

    final_corpus = matched_docs + distraction_docs[:(80 - len(matched_docs))]

    corpus_path = os.path.join(output_dir, "mini_corpus.json")
    qa_path = os.path.join(output_dir, "mini_qa.json")

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(list(final_corpus), f, ensure_ascii=False, indent=2)
        
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(list(mini_qa), f, ensure_ascii=False, indent=2)
        
    print(f"Generated mini dataset with {len(matched_docs)} matched documents and {80 - len(matched_docs)} distraction documents.")

def load_dataset():
    corpus_path = os.path.join("data", "mini_corpus.json")
    qa_path = os.path.join("data", "mini_qa.json")

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    with open(qa_path, "r", encoding="utf-8") as f:
        mini_qa = json.load(f)

    return corpus, mini_qa

if __name__ == "__main__":
    create_matched_data()