# LIDR-TB: Large Language Model-Integrated Platform for Traceable Drug Repurposing in Tuberculosis

This repository contains the core implementation and benchmark datasets for the LIDR-TB platform.
LIDR-TB (publicly available at http://lidrtb.sysbio.org.cn) is an LLM-integrated platform designed for the systematic and traceable exploration of TB repurposing knowledge. LIDR-TB integrates three core modules: a graph-structured knowledge base, an interactive network visualization engine, and a RAG-based question-answering model.


## Project Structure
- `data/`: Contains JSONL files for Single-Q, Contextual-Q, and Batch-Q benchmarks.
- `src/`: Core code for the LIDR-TB.

## Datasets
The repository includes three specialized benchmark suites:

**Single-Q**  (Basic Retrieval & Hallucination Resistance):  
Focuses on factual accuracy. It features adversarial examples (queries with no TDRKB records) to test the model's ability to report data absence instead of hallucinating. It covers both single-hop and multi-hop relational reasoning (e.g., drug-target-pathway chains).

**Contextual-Q**  (Reasoning & Coreference Resolution):  
Evaluates the model's capacity to handle multi-turn dialogues. It specifically tests coreferential expressions (e.g., "it", "this drug") and implicit entity inheritance, where the model must infer the subject from previous conversational context.

**Batch-Q**  (Parallel Extraction & Execution): 
Tests parallel processing by requiring the model to handle multiple independent requests in a single input. It evaluates multi-entity extraction, multi-intent parsing, and structural consistency in batch outputs.

## Contact
For the complete backend architecture, please contact sunshineberry@163.com.
