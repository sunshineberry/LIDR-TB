import spacy
import json, re
from typing import List, Dict, Tuple
import extract_entity as ex_entity
from llm_model import llm

# 初始化
# -----------------------------
nlp = spacy.load("en_core_sci_sm",exclude=["ner", "lemmatizer", "attribute_ruler"])
print(f"当前生效的管道组件: {nlp.pipe_names}")
# === 历史管理 ===
conversation_history=[]
def save_to_history(question: str, entities: List[Dict]):
    conversation_history.append({
        "question": question,
        "entities": entities
    })

# -----------------------------
# 规则 1：并列实体拆解（药物/对象）
# -----------------------------
def split_conj_entities_correct(question: str, nlp, entities: list = None) -> List[str]:
    """
    只对识别到的实体（如药物、目标）做并列拆解。
    entities: [{"id": "Amlodipine", "type": "Drug"}, ...]
    """
    if not entities:
        return []  # 没有实体就不拆

    doc = nlp(question)
    for token in doc:
        if token.dep_ == "conj" and token.head.dep_ in {"pobj", "dobj", "nsubj"}:
            head = token.head
            # 只拆实体
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
# 规则 2：并列名词短语共享谓词
# -----------------------------
def split_conj_noun_phrase_atomic(question: str, nlp, entities: list = None) -> List[str]:
    """
    只对识别到的实体（如药物、目标）进行并列名词拆解。
    """
    if not entities:
        return [question]  # 没有实体就不拆

    doc = nlp(question)
    entity_texts = [e["id"] for e in entities]

    for token in doc:
        if token.dep_ == "conj" and token.head.pos_ == "NOUN":
            head = token.head  # 先定义 head
            # 只有当 head 或 token 是实体时才拆
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
# 判断是否复杂句（需要 LLM 拆解）
# -----------------------------
def is_complex_sentence(question: str) -> bool:
    q_lower = question.lower()
    wh_words = {"which", "what", "who", "where", "when", "why", "how"}
    yesno_starts = ("is ", "are ", "does ", "do ", "can ", "could ", "will ", "would ")
    # 如果包含 ", and" 或 " and " 并且是 WH 或 yes/no 开头
    return (", and" in q_lower or " and " in q_lower) and (
            any(w in q_lower for w in wh_words) or q_lower.startswith(yesno_starts)
    )


def parse_llm_json_output(text: str) -> list:
    """
    将 LLM 返回的 JSON 输出提取为 Python 列表
    """
    # 提取方括号中的内容
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        # fallback 返回原句
        return [text.strip()]

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        # 去掉多余空格
        return [q.strip() for q in data if q.strip()]
    except Exception as e:
        print(f"⚠️ JSON parsing failed: {e}\nContent: {json_str}")
        return [text.strip()]


# -----------------------------
# 规则 3：yes/no + WH 并列问题 → LLM
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
    # 调用 llm_call，expect_json=True 返回 JSON 列表
    content = llm.llm_call(messages, temperature=0, max_tokens=256, expect_json=False)

    return parse_llm_json_output(content)


# -----------------------------
# 主函数：原子问题拆解
# -----------------------------
def atomic_question_decomposition(question: str) -> Tuple[List[str], List[Dict]]:
    """
    Decompose a question into atomic questions and extract entities.
    Always returns (atomic_questions, entities).
    """

    question = question.strip().rstrip("?") + "?"

    # 1️⃣ 实体抽取（只做一次）
    entities = ex_entity.extract_entities(
        question,
        history=conversation_history
    )
    # 2️⃣ 默认结果（不拆）
    atomic_questions = [question]
    # 3️⃣ 药物实体并列拆解
    res = split_conj_entities_correct(question, nlp)
    if res and len(res) > 1:
        atomic_questions = res
    # 4️⃣ 名词短语并列拆解
    elif (res := split_conj_noun_phrase_atomic(question, nlp)) and len(res) > 1:
        atomic_questions = res
    # 5️⃣ 复杂句 → LLM 拆解
    elif is_complex_sentence(question):
        atomic_questions = split_yesno_wh_with_llm(question)
    # 6️⃣ 统一保存历史（一次且必然）
    save_to_history(question, entities)

    return atomic_questions, entities

# === 从文件读取问题列表 ===
def load_queries_from_file(file_path: str):
    """
    读取 JSON 文件，提取每个 entry 的 question
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # data["entries"] 是列表，每个 entry 都有 "question"
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

# -----------------------------
# examples
# -----------------------------
if __name__ == "__main__":
    # input_file = "./test_sets/Single-Q.json"  
    # test_questions = load_queries_from_file(input_file)
    # save_atomic_questions_to_json(test_questions, "expanded_questions1.json")
    test_questions = [
        "What is the approved indication of Flupirtine, and is this drug currently approved or still in development?",
    ]

    for q in test_questions:
        print(f"\nInput: {q}")
        questions,entities= atomic_question_decomposition(q, nlp)
        print(questions)
        print(entities)

