import os
import re,yaml
import logging
from typing import List, Dict
from neo4j import GraphDatabase

with open("config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# === Neo4j Configurations ===
NEO4J_URI = CONFIG["neo4j"]["uri"]
NEO4J_USER = CONFIG["neo4j"]["user"]
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class Neo4jKnowledgeBase:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _run_query(self, query: str, **params) -> List[Dict]:
        """Execute query uniformly and return the result list"""
        try:
            with self.driver.session() as session:
                return list(session.run(query, **params))
        except Exception as e:
            logging.warning(f"Query execution failed: {e}")
            return []

    # === Internal method: convert node attributes to lowercase ===
    @staticmethod
    def _lower_props(node: Dict) -> Dict:
        return {k.lower(): v for k, v in node.items()}

    # === Internal method: format references ===
    def _format_references(self, ref_ids: List[str]) -> List[Dict]:
        ref_map = self.get_references_by_ids(ref_ids)
        return [
            {
                "Ref_ID": rid,
                "Title": info.get("title"),
                "PMID": info.get("pmid"),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{info.get('pmid')}/"
            }
            for rid, info in ref_map.items()
        ]

    # === Obtain all nodes ===
    def get_all_nodes(self, labels: List[str] = ["Drug", "Target", "DSTest", "RepurposingExp"]) -> (List[str], List[str]):
        """Return a list of UUIDs and text representations for all nodes"""
        nodes, uuids = [], []
        for label in labels:
            query = f"MATCH (n:{label}) RETURN n.uuid AS uuid, properties(n) AS props"
            for rec in self._run_query(query):
                uuid = rec.get("uuid")
                props = rec.get("props") or {}
                text = " ".join(str(v) for v in props.values() if v)
                nodes.append(text)
                uuids.append(uuid)
        return uuids, nodes

    # === Find drugs in the query ===
    def find_drugs_in_query(self, query: str) -> List[str]:
        """Extract potential drugs from the query and match against the database"""
        words = re.findall(r'[A-Za-z0-9\-]+', query)
        if not words:
            return ["unknown"]
        cypher = """
        MATCH (d:Drug)
        WHERE any(w IN $words WHERE toLower(d.Drug_name) CONTAINS toLower(w))
        RETURN d.Drug_name AS name
        """
        records = self._run_query(cypher, words=words)
        return list({r["name"] for r in records})

    # === Retrive drug information ===
    def search_drug_info(self, drug_name: str) -> list:
        """
        Return a list of drug info dictionaries with all keys in lowercase. 
        Supports case-insensitive matching for drug_name. 
        Returns [{}] if no results are found.
        """
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.Drug_name) = toLower($drug_name)
        RETURN d
        """
        result_list = []

        for rec in self._run_query(query, drug_name=drug_name):
            node = rec["d"]
            # Automatically convert all keys to lowercase
            mapped = {k.lower(): v for k, v in node.items()}
            result_list.append(mapped)

        if not result_list:
            result_list = [{}]
        return result_list

    # === Retrive drug-target ===
    def search_drug_targets(self, drug_name: str) -> Dict:
        targets = []
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.Drug_name) = toLower($name)
        MATCH (d)-[r:TARGETS]->(t:Target)
        RETURN t, r.Evidence_refs AS evidence_refs
        """
        for rec in self._run_query(query, name=drug_name):
            node = self._lower_props(rec["t"])
            evidence_refs = rec.get("evidence_refs") or []
            if isinstance(evidence_refs, str):
                evidence_refs = [ref.strip() for ref in evidence_refs.split(",") if ref.strip()]

            target_item = {
                "target_id": node.get("target_id"),
                "rv_id": node.get("rv_id"),
                "product": node.get("product") or "unknown protein",
                "ncbi_geneid": node.get("ncbi_geneid"),
                "uniprot_id": node.get("uniprot_id"),
                "functions": node.get("functions"),
                "gene_name": node.get("gene_name"),
                "functional_type": node.get("functional type"),  
                "evidence_refs": evidence_refs,
                "references": self._format_references(evidence_refs)
            }

            target_item = {k.lower(): v for k, v in target_item.items()}
            targets.append(target_item)
        return targets

    # === Retrieve repurposing experimental assays ===
    def search_experiments(self, drug_name: str) -> Dict:
        experiments = []
        query = """
            MATCH (d:Drug)
            WHERE toLower(d.Drug_name) = toLower($name)
            MATCH (d)-[r:HAS_REPURPOSING_EXP]->(e:RepurposingExp)
            RETURN e, r.Evidence_refs AS evidence_refs
        """
        for rec in self._run_query(query, name=drug_name):
            node = self._lower_props(rec["e"])
            evidence_refs = rec.get("evidence_refs") or []
            if isinstance(evidence_refs, str):
                evidence_refs = [ref.strip() for ref in evidence_refs.split(",") if ref.strip()]
            experiments.append({
                "exp_id": node.get("exp_id") or "",
                "drug_resource": node.get("drug_resource") or [],
                "drug": node.get("drugs") or "",
                "experiment_effect": node.get("effects") or "",
                "experiment_type": node.get("experiment_type") or "",
                "probable_mechanisms": node.get("probable_mechanisms") or "",
                "repurposing_methods": node.get("repurposing_methods") or [],
                "therapeutic_types": node.get("therapeutic_types") or "",
                "type_of_mechanism": node.get("type_of_mechanism") or "",
                "evidence_refs": evidence_refs,
                "references": self._format_references(evidence_refs)
            })
        return {"experiments": experiments}

    # === Retrieve Drug susceptibility tests ===
    def search_dstest_experiments(self, drug_name: str) -> list:
        dstest_experiments = []
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.Drug_name) = toLower($name)
        MATCH (d)-[r:HAS_SUSCEPTIBILITY_TEST]->(e:DSTest)
        RETURN e, r.Evidence_refs AS evidence_refs
        """
        for rec in self._run_query(query, name=drug_name):
            node = self._lower_props(rec["e"])
            evidence_refs = rec.get("evidence_refs") or []
            if isinstance(evidence_refs, str):
                evidence_refs = [ref.strip() for ref in evidence_refs.split(",") if ref.strip()]
            dstest_experiments.append({
                "drug_name": node.get("drug_name"),
                "mic_value": node.get("mic_value") or "",
                "reference_strain": node.get("reference_strain") or "",
                "species": node.get("species") or "",
                "test_strain_id": node.get("test_strain_id") or "",
                "test_strain_type": node.get("test_strain_type") or "",
                "evidence_refs": evidence_refs,
                "references": self._format_references(evidence_refs)
            })

        return dstest_experiments

    # === Retrieve Pathway ===
    def search_pathways(self, entity_type: str, entity_id: str) -> List[Dict]:
        pathways = []
        if entity_type.lower() == "drug":
            query = """
            MATCH (d:Drug)-[:TARGETS]->(t:Target)-[:ASSOCIATED_WITH]->(p:Pathway)
            WHERE toLower(d.Drug_name) = toLower($id)
            RETURN p, t.Rv_id AS rv_id, t.Product AS Product, t.Gene_name AS Gene_name
            """
        elif entity_type.lower() == "target":
            query = """
            MATCH (t:Target {Rv_id:$id})-[:ASSOCIATED_WITH]->(p:Pathway)
            RETURN p, t.Rv_id AS rv_id, t.Product AS Product, t.Gene_name AS Gene_name
            """
        else:
            raise ValueError("entity_type must be 'Drug' or 'Target'")

        for rec in self._run_query(query, id=entity_id):
            node = self._lower_props(rec["p"])
            pathways.append({
                "Pathway_ID": node.get("pathway_id"),
                "kegg_pathway_id": node.get("kegg_pathway_id"),
                "pathway_name": node.get("pathway_name"),
                "pathway_class": node.get("pathway_class"),
                "description": node.get("description"),
                "gene_list": node.get("gene_list"),
                "gene_count": node.get("gene_count"),
                "kegg_url": node.get("kegg_url"),
                "map_image_url": node.get("map_image_url"),
                "ko_pathway_id": node.get("ko_pathway_id"),
                "organism": node.get("organism"),
                "Rv_id": rec.get("rv_id"),
                "Product": rec.get("Product"),
                "Gene_name": rec.get("Gene_name")
            })
        return pathways

    # === Retrieve Reference ===
    def get_references_by_ids(self, ref_ids: List[str]) -> Dict[str, Dict]:
        if not ref_ids:
            return {}
        ref_map = {}
        query = """
        MATCH (r:Reference)
        WHERE r.Ref_ID IN $ref_ids
        RETURN r.Ref_ID AS ref_id, r.PMID AS pmid, r.Title AS title
        """
        for rec in self._run_query(query, ref_ids=ref_ids):
            rid = rec.get("ref_id")
            pmid = rec.get("pmid")
            title = rec.get("title")
            if rid and pmid:
                ref_map[rid] = {"pmid": str(pmid), "title": title}
        return ref_map

    # === Retrieve SUPPORTED_BY relationships ===
    def get_supported_by_refs(self, node_id: str) -> List[str]:
        if not node_id:
            return []
        refs = []
        query = """
        MATCH (n)-[:SUPPORTED_BY]->(r:Reference)
        WHERE n.Exp_ID = $id OR n.DSTest_ID = $id
        RETURN r.Ref_ID AS ref_id
        """
        for rec in self._run_query(query, id=node_id):
            ref_id = rec.get("ref_id")
            if ref_id:
                refs.append(ref_id)
        return refs


