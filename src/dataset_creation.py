
from datasets import Dataset
import pandas as pd
import json
import os
import arxiv, os
import wikipedia

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
MAX_RESULTS = 200
os.makedirs(DATASETS_DIR, exist_ok=True)

def create_arxiv_dataset(name:str, query: str, max_results: int=100):
    # 1. Search arXiv for papers on in-memory computing
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    print("Fetching papers...")

    papers = []
    for result in client.results(search):
        paper = {
            "title": result.title,
            "abstract": result.summary.replace("\n", " ").strip(),
            "authors": ", ".join([a.name for a in result.authors]),
            "published": result.published.strftime("%Y-%m-%d"),
            "pdf_url": result.pdf_url,
        }
        papers.append(paper)

    print(f"Fetched {len(papers)} papers.")

    OUTPUT_DIR = os.path.join(DATASETS_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------- SAVE TO JSON --------
    json_path = os.path.join(OUTPUT_DIR, f"{name}_dataset.json")
    with open(json_path, "w") as f:
        json.dump(papers, f, indent=4)
    print(f"JSON saved to {json_path}")

    # -------- ALSO SAVE AS TXT CORPUS FOR PERPLEXITY --------
    # one document per line: [TITLE] + [ABSTRACT]

    txt_path = os.path.join(OUTPUT_DIR, f"{name}_corpus.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in papers:
            doc = f"{p['title']}. {p['abstract']}"
            f.write(doc + "\n\n")

    print(f"Text corpus saved to {txt_path}")

    texts = [f"{p['title']}. {p['abstract']}" for p in papers]

    dataset = Dataset.from_dict({
        "text": texts,
        "title": [p["title"] for p in papers],
        "abstract": [p["abstract"] for p in papers],
        "authors": [p["authors"] for p in papers],
        "published": [p["published"] for p in papers],
    })
    dataset.save_to_disk(os.path.join(OUTPUT_DIR, f"{name}_dataset"))

def create_wiki_corpus(seed_topics: list[str], name: str, max_per_topic: int = 5):
    """Fetch Wikipedia articles for perplexity evaluation."""
    documents = []
    seen = set()
    
    for topic in seed_topics:
        try:
            # Search for related pages
            search_results = wikipedia.search(topic, results=max_per_topic)
            
            for title in search_results:
                if title in seen:
                    continue
                seen.add(title)
                
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    documents.append({
                        "title": page.title,
                        "text": page.content,
                        "url": page.url,
                    })
                    print(f"Fetched: {page.title}")
                except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
                    print(f"Skipping {title}: {e}")
                    
        except Exception as e:
            print(f"Search failed for {topic}: {e}")
    
    OUTPUT_DIR = os.path.join(DATASETS_DIR, name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------- SAVE TO JSON --------
    json_path = os.path.join(OUTPUT_DIR, f"{name}_dataset.json")
    with open(json_path, "w") as f:
        json.dump(documents, f, indent=4)
    print(f"JSON saved to {json_path}")

    # -------- ALSO SAVE AS TXT CORPUS FOR PERPLEXITY --------
    # one document per line: [TITLE] + [ABSTRACT]

    txt_path = os.path.join(OUTPUT_DIR, f"{name}_corpus.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in documents:
            doc = f"{p['title']}. {p['text']}"
            f.write(doc + "\n\n")

    print(f"Text corpus saved to {txt_path}")

    texts = [f"{p['title']}. {p['text']}" for p in documents]

    dataset = Dataset.from_dict({
        "text": texts,
        "title": [p["title"] for p in documents],
    })
    dataset.save_to_disk(os.path.join(OUTPUT_DIR, f"{name}_dataset"))

if __name__ == "__main__":
    dataset_name = "in_memory_computing"
    # Query for topic
    query = (
    '(cat:cs.AR) AND '
    '("in-memory computing" OR "processing in memory" OR "compute in memory")'
    )
    create_arxiv_dataset(name=dataset_name, query=query, max_results=MAX_RESULTS)

    # Example usage for different domains
    # Food
    create_wiki_corpus(
        seed_topics=["Food", "Cuisine", "Cooking"],
        name="food_corpus"
    )

    # Celebrities/Film
    create_wiki_corpus(
        seed_topics=["Anne Hathaway", "Anne Hathaway filmography", "The Dark Knight Rises", 
                    "Anne Hathaway actresses", "Anne Hathaway hollywood"],
        name="anne_corpus"
    )