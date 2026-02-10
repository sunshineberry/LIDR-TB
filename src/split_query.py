import spacy
import json, re
from typing import List, Dict, Tuple
import extract_entity as ex_entity
from llm_model import llm

# Load spacy model
nlp = spacy.load("en_core_sci_sm",exclude=["ner", "lemmatizer", "attribute_ruler"])

conversation_history=[]
def save_to_history(question: str, entities: List[Dict]):
    conversation_history.append({
        "question": question,
        "entities": entities
    })

# -----------------------------
# Rule 1: Decomposition of coordinate entities (Drugs/Objects)
# -----------------------------
def split_conj_entities_correct(question: str, nlp, entities: list = None) -> List[str]:
    """
    Decompose coordinate entities (e.g., Drugs, Targets) only.
    entities: [{"id": "Amlodipine", "type": "Drug"}, ...]
    """
    if not entities:
        return []  
    doc = nlp(question)
    for token in doc:
        if token.dep_ == "conj" and token.head.dep_ in {"pobj", "dobj", "nsubj"}:
            head = token.head
            if head.text not in [e["id"] for e in entities] or token.text not in [e["id"] for e in entities]:
                continue
            prefix_tokens = [t.text for t in doc[:head.i]]
            q1 = " ".join(prefix_tokens + [head.text]).strip()
            q2 = " ".join(prefix_tokens + [token.text]).strip()
            if not q1.endswith("?"):
                q1 += "?"
            if not q2.endswith("?"):
                q2 += "?"
            return [q1, q2]
    return []

# -----------------------------
# Rule 2: Coordinate noun phrases with shared predicates.
# -----------------------------
def split_conj_noun_phrase_atomic(question: str, nlp, entities: list = None) -> List[str]:
    """
    Perform coordinate noun decomposition only on identified entities (e.g., Drugs, Targets).。
    """
    if not entities:
        return [question]
    doc = nlp(question)
    entity_texts = [e["id"] for e in entities]

    for token in doc:
        if token.dep_ == "conj" and token.head.pos_ == "NOUN":
            head = token.head  
            if head.text not in entity_texts and token.text not in entity_texts:
                continue
            mods = [t.text for t in head.lefts if t.dep_ in {"amod", "compound"}]
            head_phrase = " ".join(mods + [head.text])
            conj_phrase = " ".join(mods + [token.text])
            start = head.idx
            end = token.idx + len(token.text)
            base = question[:start] + "{X}" + question[end:]
            return [
                base.replace("{X}", head_phrase),
                base.replace("{X}", conj_phrase),
            ]
    return [question]

# -----------------------------
# Determine if the query is a complex sentence (requires LLM decomposition)
# -----------------------------
def is_complex_sentence(question: str) -> bool:
    q_lower = question.lower()
    wh_words = {"which", "what", "who", "where", "when", "why", "how"}
    yesno_starts = ("is ", "are ", "does ", "do ", "can ", "could ", "will ", "would ")
    return (", and" in q_lower or " and " in q_lower) and (
            any(w in q_lower for w in wh_words) or q_lower.startswith(yesno_starts)
    )


def parse_llm_json_output(text: str) -> list:
    """
    Extract JSON output from LLM and parse into a Python list
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return [text.strip()]

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        return [q.strip() for q in data if q.strip()]
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}\nContent: {json_str}")
        return [text.strip()]


# -----------------------------
# Rule 3: Combined Yes/No and WH-questions -> LLM
# -----------------------------
def split_yesno_wh_with_llm(question: str) -> List[str]:
    prompt = f"""
    You are an expert question splitter.

    Task:
    - Split the input question into multiple independent questions, each representing one atomic action.
    - Keep all entities intact; do not remove or replace them.
    - Ensure each question is complete and can be understood independently.
    - Only split where there is a semantic separation between distinct actions (do not split within an entity or a verb phrase that is a single action).
    - Do NOT split if multiple noun phrases are modified by the same verb and can be understood together.

    Input question: "{question}"

    Output format:
    - Only output a valid JSON list of strings in the form:
    ["Question1?", "Question2?", ...]
    - Do NOT include explanations, bullet points, or extra text.

    Examples:
    Input: "What are the targets of Amlodipine and Acarbose?"
    Output: ["What are the targets of Amlodipine?", "What are the targets of Acarbose?"]

    Input: "Are there any tuberculosis repositioning studies involving Mefloquine, and what repurposing methods were used?"
    Output: ["Are there any tuberculosis repositioning studies involving Mefloquine?", "What repurposing methods were used for Mefloquine?"]
    """
    messages = [{"role": "user", "content": prompt}]
    # Extract JSON output from LLM and convert to a Python list
    content = llm.llm_call(messages, temperature=0, max_tokens=256, expect_json=False)

    return parse_llm_json_output(content)

def atomic_question_decomposition(question: str) -> Tuple[List[str], List[Dict]]:
    """
    Decompose a question into atomic questions and extract entities.
    Always returns (atomic_questions, entities).
    """

    question = question.strip().rstrip("?") + "?"

    # 1. entity extraction
    entities = ex_entity.extract_entities(
        question,
        history=conversation_history
    )
    # 2. Default: No decomposition
    atomic_questions = [question]
    # 3. Decompose coordinate drug entities
    res = split_conj_entities_correct(question, nlp)
    if res and len(res) > 1:
        atomic_questions = res
    # 4. Decompose coordinate noun phrases
    elif (res := split_conj_noun_phrase_atomic(question, nlp)) and len(res) > 1:
        atomic_questions = res
    # 5. Complex sentences -> LLM-based decomposition
    elif is_complex_sentence(question):
        atomic_questions = split_yesno_wh_with_llm(question)
    # 6. Unified history logging (Mandatory & Atomic)
    save_to_history(question, entities)

    return atomic_questions, entities

def load_queries_from_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = [entry.get("question", "") for entry in data.get("entries", [])]
    return queries

def save_atomic_questions_to_json(input_questions: List[str], output_file: str):
    output_data = []
    for q in input_questions:
        atomic_qs,_ = atomic_question_decomposition(q, nlp)
        output_data.append({
            "original_question": q,
            "atomic_questions": atomic_qs
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"questions": output_data}, f, indent=2, ensure_ascii=False)



