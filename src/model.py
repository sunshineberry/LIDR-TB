import time
from typing import List, Dict
from config import *
import json
from openai import OpenAI
from split_query import atomic_question_decomposition, nlp
from llm_model import llm,KB
from jinja2 import Environment, FileSystemLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

conversation_history: List[Dict] = []
MAX_HISTORY = 10
if len(conversation_history) >= MAX_HISTORY:
    conversation_history.pop(0)

env = Environment(loader=FileSystemLoader("./"))
template = env.get_template("prompt.jinja2")

render_cache = {}

def get_last_entity_of_type(entity_type="Drug") -> Dict[str, str]:
    for past in reversed(conversation_history):
        for e in past.get("entities", []):
            if e.get("type") == entity_type:
                return e
    return {}

def subintent_to_neo4j_attr(sub_intent: str) -> str:
    return SUBINTENT_TO_NEO4J.get(sub_intent.lower(), sub_intent) if sub_intent else ""
    
# === Intent Recognition ===
def detect_intent(user_query: str, query_entity) -> Dict:
    if isinstance(user_query, dict):
        user_query = user_query.get("question", "") or user_query.get("query", "") or ""
        if not isinstance(user_query, str):
            user_query = str(user_query)

    user_query = user_query.strip()
    q = user_query.lower()
    entities = query_entity
    main_entity = entities[0] if entities else get_last_entity_of_type("Drug")
    context_link = ", ".join(e["id"] for e in entities) if entities else main_entity.get("id","") if main_entity else ""

    sub_intent_rule = None
    for key, synonyms in SYNONYM_MAP.items():
        if any(s in q for s in synonyms):
            sub_intent_rule = key
            break

    fewshot = "\n".join([f"examples: {ex['question']} → {ex['intent']}" for ex in FEW_SHOT_EXAMPLES])
    system_prompt = f"""
        You are an expert tuberculosis drug QA assistant and a JSON extractor.
        Given a user query, extract intent and sub_intent in JSON format.
        Follow these rules:
        - If the query contains MIC, reference strains, or sensitivity data → Drug Sensitivity Test
        - If the query mentions targets, genes, or proteins → Target
        - If the query mentions experiments or drug repositioning → Drug Repurposing Assay
        - If the query mentions pathways, biological pathways, or mechanisms → Pathway
        - Otherwise, classify as Drug Information or Unknown
        Examples: {fewshot}
        """
    messages = [{"role":"system","content":system_prompt}, {"role":"user","content":user_query}]
    llm_res = llm.llm_call(messages, temperature=0.1, max_tokens=256, response_format={"type":"json_object"}, expect_json=True)

    sub_intent_final = sub_intent_rule or llm_res.get("sub_intent")
    intent_final = llm_res.get("intent") or "Drug Information"

    default_subintent = {"Target": "target", "Pathway": "pathway", "Drug Information": "basic_info"}
    if not sub_intent_final:
        sub_intent_final = default_subintent.get(intent_final, "default")

    return {
        "intent": intent_final,
        "sub_intent": sub_intent_final,
        "entities": [main_entity] if main_entity else [],
        "context_link": context_link
    }

def save_to_history(question: str, entities: List[Dict], intent_json: Dict):
    conversation_history.append({
        "question": question,
        "entities": entities,
        "intent": intent_json.get("intent"),
        "sub_intent": intent_json.get("sub_intent")
    })

# === KB field mapping and data processing ===
key_fields_map = {
    "drug information": {
        "name": "drug_name",
        "indication": "indication",
        "approval_stage": "stages"
    },
    "drug sensitivity test": {
        "mic value": "mic_value",
        "reference strains": "reference_strain",
        "pathogen species": "species",
        "test strain id": "test_strain_id",
        "test strain type": "test_str_type"
    }
}

def get_kb_field(intent_key: str, sub_intent_key: str) -> str:
    intent_key = intent_key.lower()
    sub_intent_key = sub_intent_key.lower()
    return key_fields_map.get(intent_key, {}).get(sub_intent_key, "default")

def extract_fields(kb_result: list, field_key: str):
    if not kb_result:
        return []
    if field_key == "default":
        return kb_result
    return [item.get(field_key, "") for item in kb_result]

def process_pathway_results(raw_results):
    simplified = {}
    for item in raw_results:
        pid = item.get("Pathway_ID")
        pname = item.get("pathway_name")
        pclass = item.get("pathway_class")
        gene = item.get("Gene_name")
        rv_id = item.get("Rv_id")

        if pid not in simplified:
            simplified[pid] = {"pathway_name": pname, "pathway_class": pclass, "genes": set()}
        if gene and gene != "/":
            simplified[pid]["genes"].add(gene)
        if rv_id and rv_id != "/":
            simplified[pid]["genes"].add(rv_id)

    result = []
    for pid, info in simplified.items():
        result.append({
            "Pathway_ID": pid,
            "pathway_name": info["pathway_name"],
            "pathway_class": info["pathway_class"],
            "genes": list(info["genes"])
        })
    return result

# === Prompt templates ===
def render_prompt_cached(title, intent_data, intent_name):
    key = (title, json.dumps(intent_data, sort_keys=True))
    if key in render_cache:
        return render_cache[key]
    rendered = template.render(
        title=title,
        structured_data=intent_data,
        intent=intent_name.title(),
        content_text=json.dumps(intent_data, ensure_ascii=False)
    )
    render_cache[key] = rendered
    return rendered

def process_query(user_query: str):
    atomic_queries, query_entities = atomic_question_decomposition(user_query, nlp)

    # If only has one entity
    if len(query_entities) == 1:
        query_entities = query_entities * len(atomic_queries)
    intent_method_map = {
        "drug information": lambda e: KB.search_drug_info(e.get("id")),
        "target": lambda e: KB.search_drug_targets(e.get("id")),
        "pathway": lambda e: KB.search_pathways(e.get("type"), e.get("id")),
        "drug repurposing assay": lambda e: KB.search_experiments(e.get("id")),
        "drug sensitivity test": lambda e: KB.search_dstest_experiments(e.get("id"))
    }
    intent_data_list = []


    for aq, entity_info in zip(atomic_queries, query_entities):
        intent_json = detect_intent(aq, [entity_info])
        save_to_history(aq, intent_json.get("entities", []), intent_json)

        intent_key = intent_json.get("intent", "").lower()
        # print(intent_key)
        sub_intent_key = str(intent_json.get("sub_intent", "default")).lower()
        # print(sub_intent_key)
        field_key = get_kb_field(intent_key, sub_intent_key)
        if intent_key =="target" and sub_intent_key == "pathway":
            search_method = intent_method_map.get(sub_intent_key)
        else:
            search_method = intent_method_map.get(intent_key)
        if not search_method:
            intent_data_list.append({
                "question": aq,
                "intent": intent_key,
                "sub_intent": sub_intent_key,
            })
            continue
        structured_data = []
        for entity in intent_json.get("entities", []):
            kb_result = search_method(entity)
            extracted = extract_fields(kb_result, field_key)
            structured_data.extend(extracted if isinstance(extracted, list) else [extracted])
            # 特殊处理通路结果
            if intent_key == "pathway":
                structured_data = process_pathway_results(structured_data)
        # print(structured_data)

        intent_data_list.append({
            "question": aq,
            "intent": intent_key,
            "sub_intent": sub_intent_key,
            "entity_name": entity.get("id"),
            "data": structured_data
        })

    prompt_text = render_prompt_cached(user_query, intent_data_list, intent_key)
    messages = [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": "Please answer the above questions based on the structured data."}
    ]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS
    )
    return response.choices[0].message.content.strip()

def batch_test(queries):
    results = {}
    for q in queries:
        results[q] = process_query(q)
    return results

def process_entries_json(input_file, output_file):

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", [])
    if not entries:
        raise ValueError("No entries found in input JSON.")

    # Extract questions
    questions = [e["question"] for e in entries]

    batch_results = batch_test(questions)

    # Attach answers
    for entry in entries:
        q = entry["question"]
        entry["answer"] = batch_results.get(q, "")
        print("=" * 80)
        print(entry["answer"])

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=4)

    # print(f"Answers saved to {output_file}")


if __name__ == "__main__":
    queries = ["What is the MIC value of ebselen, and which reference strain was used in the experiment?",
               "What is the approved indication of Glimepiride, and is this drug currently approved or still in development?",
               "Are there any tuberculosis repositioning studies involving bromfenac, and what repurposing methods were used?"]

    batch_results = batch_test(queries)
    for q, english_answer in batch_results.items():
        print(q)
        print(english_answer)
        print("=" * 80)




