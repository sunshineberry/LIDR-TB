import os
import yaml
from dotenv import load_dotenv

load_dotenv()

# === Load configure files ===
with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# === Neo4j Configurations ===
NEO4J_URI = CONFIG["neo4j"]["uri"]
NEO4J_USER = CONFIG["neo4j"]["user"]
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# === LLM Configurations ===
LLM_MODEL = CONFIG["llm"]["model"]
LLM_TEMPERATURE = CONFIG["llm"]["temperature"]
LLM_MAX_TOKENS = CONFIG["llm"]["max_tokens"]

LLM_BASE_URL = CONFIG["llm"]["base_url"]
LLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not LLM_API_KEY:
    raise EnvironmentError(
        "The environment variable DASHSCOPE_API_KEY is not set. Please configure your DashScope API key in the system environment."
    )

# === （few-shot） ===
FEW_SHOT_EXAMPLES = [
    {"question": "What are the repurposing experiments for Ipragliflozin?", "intent": "Drug Repurposing Assay"},
    {"question": "What are its targets?", "intent": "Target"},
    {"question": "What is the reference strain information?", "intent": "Drug Sensitivity Test"}
]

# === Synonym normalization ===
SYNONYM_MAP = {
    "indication": ["indication", "indications", "used for", "therapeutic use", "treats", "indicated for"],
    "ATC_code": ["atc", "atc code", "atc codes", "anatomical therapeutic chemical code"],
    "approval_stage": ["approval", "stage", "regulatory status", "clinical stage","approval stage","approval status"],
    "function": ["function", "role", "biological function", "activity"],
    "protein": ["protein", "protein product", "gene product"],
    "description": ["description", "summary", "overview", "pathway description", "pathway summary"],
    "gene_count": ["gene count", "number of genes", "genes involved", "members"],
    "pathway_class": ["pathway class", "type", "category", "pathway type"],
}

SUBINTENT_TO_NEO4J = {
    "indication": "Indication",
    "ATC_code": "ATC_code",
    "approval_stage": "Stages",
    "function": "Functions",
    "protein": "Product",
    "description": "description",
    "gene_count": "gene_count",
    "pathway_class": "pathway_class",
}

KEYWORD_BLACKLIST = {
    "target","targets","pathway","pathways","gene","genes","protein","proteins",
    "mechanism","mechanisms","mic","indication","indications"
}

