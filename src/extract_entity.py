from typing import List, Dict
from config import *
import spacy
from collections import OrderedDict
import re
import unicodedata
from functools import lru_cache
from llm_model import KB

# 加载 SciSpaCy 化学实体模型
nlp_chemical = spacy.load("en_ner_bc5cdr_sm",exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])
# === 工具函数 ===
# 文本标准化
def _normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s).lower()
    if s.endswith("s") and len(s) > 3:
        s = s[:-1]
    return s

# 预编译正则并缓存
@lru_cache(maxsize=512)
def _compile_word_boundary_pattern(candidate: str):
    cand_norm = _normalize_text(candidate)
    # \b 匹配完整单词边界，忽略大小写
    return re.compile(rf"\b{re.escape(cand_norm)}\b", re.IGNORECASE)

def _word_boundary_match(query: str, candidate: str) -> bool:
    if not candidate:
        return False
    query_norm = _normalize_text(query)
    pattern = _compile_word_boundary_pattern(candidate)
    return pattern.search(query_norm) is not None


def get_last_entity_of_type(entity_type="Drug", history=None) -> Dict[str, str]:
    history = history or []  # 默认空列表
    for past in reversed(history):
        for e in past.get("entities", []):
            if e.get("type") == entity_type:
                return e
    return {}

# === 实体抽取 ===
def extract_entities(query: str, context_entities=None, use_history=True,history=None) -> List[Dict[str, str]]:
    """
    从单句或原子问题中抽取实体，并结合历史上下文回退。
    使用 SciSpaCy en_ner_bc5cdr_md 模型识别化学/药物实体。
    支持连续问句回退到上一条上下文实体。
    """
    context_entities = context_entities or []
    history = history or []  # 默认空列表

    query_norm = _normalize_text(query)
    entity_set = OrderedDict()

    # ------------------------------
    # 1. SciSpaCy 化学/药物实体识别
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
    # 2. 上下文 / 历史回退：代词或上下文实体处理
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
    # 3. 如果仍未识别，查询知识库匹配药物
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
    # 4. 最终兜底：unknown
    # ------------------------------
    if not entity_set:
        entity_set["__unknown_entity__"] = {"id": "unknown", "type": "unknown"}

    return list(entity_set.values())
