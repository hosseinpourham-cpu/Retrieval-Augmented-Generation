import requests
import os
import config 

SAVE_PATH = os.path.join(config.DATA_PATH,"corpus.txt")
MAX_SIZE_MB = 50

wiki_pages = [
    "Machine_learning",
    "neural_network",
    "Deep_learning",
    "Gradient_descent",
    "Overfitting",
    "Backpropagation",
    "Randomized_algorithm",
    "Stochastic_gradient_descent",
    "Convex_optimization",
    "Loss_functions_for_classification",
    "Cost_function",
    "Gradient_clipping",
    "Feature_engineering",
    "Training_data",
    "Test_data",
    "Data_preprocessing",
    "Feedforward_neural_network",
    "Multilayer_perceptron",
    "Dropout_(neural_networks)",
    "Batch_normalization",
    "Weight_initialization",
    "Long_short-term_memory",
    "Gated_recurrent_unit",
    "Autoencoder",
    "Generative_adversarial_network",
    "Vision_transformer",
    "Large_language_model",
    "Attention_(machine_learning)",
    "Self-attention",
    "Prompt_engineering",
    "Transfer_learning",
    "Curse_of_dimensionality",
    "Overparameterization",
    "Statistical_learning_theory",
]

def fetch_wikipedia(title):
    url = "https://en.wikipedia.org/w/api.php"
    
    headers = {
        "User-Agent": "RAG-System/1.0 (your_email@example.com)"
    }

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": True,
        "titles": title
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)

        if response.status_code != 200:
            print(f" Failed request for {title} (status {response.status_code})")
            return ""

        data = response.json()

        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract", "")
            print(f"✅ {title}: {len(text)} chars")
            return text

    except Exception as e:
        print(f"Error fetching {title}: {e}")

    return ""


                          

def fetch_arxiv(query="machine learning", max_results=500):
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(url)
    return response.text


def build_corpus():
    os.makedirs("./data/docs", exist_ok=True)

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        for page in wiki_pages:
            print(f"Fetching {page}")
            text = fetch_wikipedia(page)

            if text:
                f.write(text + "\n\n")
        print("Fetching arXiv...")
        arxiv_text = fetch_arxiv()

        if arxiv_text:
            f.write(arxiv_text)        

    size_mb = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    print(f"\nFinal corpus size: {size_mb:.2f} MB")


build_corpus()
