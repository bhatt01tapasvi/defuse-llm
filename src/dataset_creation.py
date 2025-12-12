from datasets import Dataset
import json
import time
import re
import os
import requests
import arxiv
import wikipedia
import mwparserfromhell

wikipedia.set_lang("en")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
MAX_RESULTS = 200
os.makedirs(DATASETS_DIR, exist_ok=True)

# Optional rate limit to avoid Wikipedia throttling
REQUEST_DELAY = 1.5  # Increase to 1.5 seconds
MAX_RETRIES = 3

# User-Agent is REQUIRED by Wikipedia API
HEADERS = {
    'User-Agent': 'WikiCorpusBuilder/1.0 (Educational Research Project; Contact: your-email@example.com)'
}

EXISTING_CUSTOM_DATASETS = [
    "in_memory_computing_corpus", 
    "food_corpus", 
    "anne_corpus",
    "college_computer_science_corpus",
    "abstract_algebra_corpus",
    "high_school_biology_corpus",
    "high_school_world_history_corpus",
    "marketing_corpus",
    "philosophy_corpus",
    "professional_law_corpus"
    ]

def create_arxiv_dataset(name:str, query: str, max_results: int=100):
    """Create an arXiv dataset from a search query."""
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

# -------------------------------------------------------------------
# Fetch raw WikiText
# -------------------------------------------------------------------

def fetch_wikitext_article(title: str, retries=MAX_RETRIES):
    """Fetch raw WikiText markup from Wikipedia Page API."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "prop": "revisions",
        "rvprop": "content",
        "format": "json",
        "formatversion": 2
    }

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            
            # Check if response is empty
            if not resp.text:
                print(f"   ✗ Empty response (attempt {attempt + 1}/{retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
                
            data = resp.json()
            break
            
        except requests.exceptions.JSONDecodeError as e:
            print(f"   ✗ JSON decode error (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                continue
            return None
        except Exception as e:
            print(f"   ✗ Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None
    else:
        print(f"   ✗ All {retries} attempts failed")
        return None

    try:
        page = data["query"]["pages"][0]

        # Skip missing pages
        if "missing" in page:
            return None
            
        # Skip redirect pages
        if "redirect" in page:
            return None

        # Check if revisions exist
        if "revisions" not in page or not page["revisions"]:
            return None

        content = page["revisions"][0]["content"]
        
        # Skip redirect pages by checking content
        if "#REDIRECT" in content or "#redirect" in content:
            return None
            
        return content
        
    except (KeyError, IndexError) as e:
        print(f"   ✗ Parse error: {e}")
        return None


# -------------------------------------------------------------------
# Clean WikiText → WikiText-103 style (light cleaning)
# -------------------------------------------------------------------

def clean_wikitext(wikitext: str) -> str:
    """Light cleaning to mimic WikiText-103 format."""
    text = wikitext

    # Remove <ref>...</ref> but preserve paragraph structure
    text = re.sub(r"<ref[^>/]*>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref[^>/]*/>", "", text)

    # Remove <references> blocks
    text = re.sub(r"<references.*?>.*?</references>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Drop file/image links
    text = re.sub(r"\[\[(File|Image):.*?\]\]", "", text, flags=re.IGNORECASE)

    # Remove categories
    text = re.sub(r"\[\[Category:.*?\]\]", "", text, flags=re.IGNORECASE)

    # Remove comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Internal links: keep label
    text = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    # Remove external link URLs, keep label
    text = re.sub(r"\[https?://[^\s\]]+\s+([^\]]+)\]", r"\1", text)

    # Remove bare external links
    text = re.sub(r"\[https?://[^\s\]]+\]", "", text)

    # Remove any HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Remove tables
    text = re.sub(r"\{\|.*?\|\}", "", text, flags=re.DOTALL)

    # Normalize headers to WikiText format: `== Header ==` → `= = Header = =`
    def header_repl(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        return "\n" + ("= " * level) + title + (" =" * level) + "\n"

    text = re.sub(r"(={2,6})\s*([^=\n]+?)\s*\1", header_repl, text)

    # Just trim trailing spaces per line, preserve all newlines
    text = "\n".join([line.rstrip() for line in text.split("\n")])

    return text.strip()


# -------------------------------------------------------------------
# Corpus Builder - SAVES IN YOUR ORIGINAL FORMAT
# -------------------------------------------------------------------

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
                    print(f"  Fetching: {title}")
                    time.sleep(REQUEST_DELAY)
                    
                    # Fetch raw wikitext and clean it
                    raw = fetch_wikitext_article(title)
                    if not raw:
                        print(f"   ✗ Could not fetch wikitext")
                        continue
                    
                    cleaned = clean_wikitext(raw)
                    if not cleaned:
                        print("   ✗ Empty after cleaning")
                        continue
                    
                    # Get URL from wikipedia library
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    documents.append({
                        "title": page.title,
                        "text": cleaned,
                        "url": page.url,
                    })
                    print(f"   ✓ Fetched: {page.title}")
                    
                except (wikipedia.DisambiguationError, wikipedia.PageError) as e:
                    print(f"   ✗ Skipping {title}: {e}")
                    
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
    # WikiText format: raw text without prepending title
    txt_path = os.path.join(OUTPUT_DIR, f"{name}_corpus.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in documents:
            f.write(p["text"] + "\n\n")

    print(f"Text corpus saved to {txt_path}")

    texts = [p["text"] for p in documents]

    dataset = Dataset.from_dict({
        "text": texts,
        "title": [p["title"] for p in documents],
    })
    dataset.save_to_disk(os.path.join(OUTPUT_DIR, f"{name}_dataset"))

# ============================================================================
# SPECIFIC DATASET CREATION FUNCTIONS
# ============================================================================

def create_in_memory_computing_dataset():
    """Create dataset about in-memory computing from arXiv."""
    dataset_name = "in_memory_computing"
    query = (
        '(cat:cs.AR) AND '
        '("in-memory computing" OR "processing in memory" OR "compute in memory")'
    )
    print(f"\n{'='*80}")
    print(f"Creating {dataset_name} dataset from arXiv")
    print(f"Query: {query}")
    print(f"{'='*80}\n")
    create_arxiv_dataset(name=dataset_name, query=query, max_results=MAX_RESULTS)


def create_food_corpus_dataset():
    """Create dataset about food and cooking from Wikipedia."""
    name = "food_corpus"
    seed_topics = ["Food", "Cuisine", "Cooking"]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_anne_hathaway_corpus_dataset():
    """Create dataset about Anne Hathaway from Wikipedia."""
    name = "anne_corpus"
    seed_topics = [
        "Anne Hathaway", 
        "Anne Hathaway filmography", 
        "The Dark Knight Rises", 
        "Anne Hathaway actresses", 
        "Anne Hathaway hollywood"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )
 

# ============================================================================
# SPECIFIC DATASET CREATION FUNCTIONS - MMLU-based Wikipedia Corpora
# ============================================================================

def create_college_computer_science_corpus_dataset():
    """Create dataset about college computer science from Wikipedia."""
    name = "college_computer_science_corpus"
    seed_topics = [
        "Computer Science",
        "Algorithm",
        "Data Structure",
        "Computer Architecture",
        "Operating System",
        "Database",
        "Computer Network",
        "Theory of Computation"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_abstract_algebra_corpus_dataset():
    """Create dataset about abstract algebra from Wikipedia."""
    name = "abstract_algebra_corpus"
    seed_topics = [
        "Abstract Algebra",
        "Group Theory",
        "Ring Theory",
        "Field Theory",
        "Module",
        "Vector Space",
        "Linear Algebra",
        "Galois Theory"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_high_school_biology_corpus_dataset():
    """Create dataset about high school biology from Wikipedia."""
    name = "high_school_biology_corpus"
    seed_topics = [
        "Biology",
        "Cell Biology",
        "Genetics",
        "Evolution",
        "Ecology",
        "Anatomy",
        "Physiology",
        "Botany",
        "Zoology"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_high_school_world_history_corpus_dataset():
    """Create dataset about high school world history from Wikipedia."""
    name = "high_school_world_history_corpus"
    seed_topics = [
        "World History",
        "Ancient Civilization",
        "Medieval History",
        "Renaissance",
        "Industrial Revolution",
        "World War I",
        "World War II",
        "Cold War",
        "Modern History"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_marketing_corpus_dataset():
    """Create dataset about marketing from Wikipedia."""
    name = "marketing_corpus"
    seed_topics = [
        "Marketing",
        "Brand Management",
        "Consumer Behavior",
        "Market Research",
        "Digital Marketing",
        "Advertising",
        "Public Relations",
        "Sales",
        "Marketing Strategy"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_philosophy_corpus_dataset():
    """Create dataset about philosophy from Wikipedia."""
    name = "philosophy_corpus"
    seed_topics = [
        "Philosophy",
        "Ethics",
        "Metaphysics",
        "Epistemology",
        "Logic",
        "Political Philosophy",
        "Philosophy of Mind",
        "Aesthetics",
        "Ancient Philosophy",
        "Modern Philosophy"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


def create_professional_law_corpus_dataset():
    """Create dataset about professional law from Wikipedia."""
    name = "professional_law_corpus"
    seed_topics = [
        "Law",
        "Constitutional Law",
        "Contract Law",
        "Criminal Law",
        "Tort Law",
        "Property Law",
        "Administrative Law",
        "Civil Procedure",
        "Legal Ethics",
        "Evidence Law"
    ]
    
    print(f"\n{'='*80}")
    print(f"Creating {name} dataset from Wikipedia")
    print(f"Seed topics: {seed_topics}")
    print(f"{'='*80}\n")
    
    create_wiki_corpus(
        seed_topics=seed_topics,
        name=name,
        max_per_topic=5
    )


# ============================================================================
# BATCH CREATION FUNCTIONS
# ============================================================================

def create_all_datasets():
    """Create all predefined datasets."""
    print("\n" + "="*80)
    print("CREATING ALL DATASETS")
    print("="*80 + "\n")
    
    #create_in_memory_computing_dataset()
    create_food_corpus_dataset()
    create_anne_hathaway_corpus_dataset()

    create_college_computer_science_corpus_dataset()
    create_abstract_algebra_corpus_dataset()
    create_high_school_biology_corpus_dataset()
    create_high_school_world_history_corpus_dataset()
    create_marketing_corpus_dataset()
    create_philosophy_corpus_dataset()
    create_professional_law_corpus_dataset()
    
    print("\n" + "="*80)
    print("ALL DATASETS CREATED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Option 1: Create all datasets
    create_all_datasets()
    
    # Option 2: Create specific datasets individually
    # create_in_memory_computing_dataset()
    # create_food_corpus_dataset()
    # create_anne_hathaway_corpus_dataset()