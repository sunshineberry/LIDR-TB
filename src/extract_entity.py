from typing import List, Dict
from config import *
import spacy
from collections import OrderedDict
import re
import unicodedata
from functools import lru_cache
from llm_model import KB

# Load SciSpaCy Chemical entity model
nlp_chemical = spacy.load("en_ner_bc5cdr_sm",exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])

# Text Normalization
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s).lower()
    if s.endswith("s") and len(s) > 3:
        s = s[:-1]
    return s

@lru_cache(maxsize=512)
def _compile_word_boundary_pattern(candidate: str):
    cand_norm = _normalize_text(candidate)
    return re.compile(rf"\b{re.escape(cand_norm)}\b", re.IGNORECASE)

def _word_boundary_match(query: str, candidate: str) -> bool:
    if not candidate:
        return False
    query_norm = _normalize_text(query)
    pattern = _compile_word_boundary_pattern(candidate)
    return pattern.search(query_norm) is not None


def get_last_entity_of_type(entity_type="Drug", history=None) -> Dict[str, str]:
    history = history or []  
    for past in reversed(history):
        for e in past.get("entities", []):
            if e.get("type") == entity_type:
                return e
    return {}

# === Entity extraction ===
def extract_entities(query: str, context_entities=None, use_history=True,history=None) -> List[Dict[str, str]]:
    context_entities = context_entities or []
    history = history or [] 

    query_norm = _normalize_text(query)
    entity_set = OrderedDict()

    # ------------------------------
    # 1. SciSpaCy entity recognition
    # ------------------------------
    try:
        doc = nlp_chemical(query)
        candidates = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
    except Exception as e:
        print(f"⚠️ SciSpaCy entity identification failure: {e}")
        candidates = []

    for cand in candidates:
        cand_norm = _normalize_text(cand)
        if cand_norm.lower() in KEYWORD_BLACKLIST:
            continue
        if _word_boundary_match(query_norm, cand_norm):
            entity_set[cand_norm.lower()] = {"id": cand_norm, "type": "Drug"}

    # --------------------------------------------
    # 2. conversational history and coreference resolution
    # --------------------------------------------
    pronouns = {"it", "they", "this", "that", "these", "those", "them", "its", "this drug", "drug"}
    has_pronoun = any(tok in pronouns for tok in query_norm.split())

    if not entity_set and (has_pronoun or not candidates):
        for e in context_entities:
            key = _normalize_text(e["id"]).lower()
            entity_set.setdefault(key, {"id": e["id"], "type": e.get("type", "Drug")})

        if not entity_set and use_history:
            chosen = get_last_entity_of_type("Drug", history=history) or get_last_entity_of_type(history=history)
            if chosen:
                key = _normalize_text(chosen["id"]).lower()
                entity_set.setdefault(key, {"id": chosen["id"], "type": chosen.get("type", "Drug")})

    # ---------------------------------------
    # 3. If still unidentified, query the knowledge base for drug matching
    # ---------------------------------------
    if not entity_set:
        try:
            candidates = KB.find_drugs_in_query(query) or []
        except Exception as e:
            print(f"⚠️ KB query failure: {e}")
            candidates = []

        for cand in candidates:
            cand_norm = _normalize_text(cand)
            if cand_norm.lower() in KEYWORD_BLACKLIST:
                continue
            if _word_boundary_match(query_norm, cand_norm):
                entity_set[cand_norm.lower()] = {"id": cand_norm, "type": "Drug"}

    # ------------------------------
    # 4. Finally：unknown
    # ------------------------------
    if not entity_set:
        entity_set["__unknown_entity__"] = {"id": "unknown", "type": "unknown"}

    return list(entity_set.values())

