import os
import re
import json
import faiss
import sqlite3
import numpy as np
import pandas as pd
import datetime, csv, html
from dotenv import load_dotenv
from dataclasses import dataclass
from openai import AzureOpenAI, OpenAI
from typing import List, Tuple, Dict, Any, Optional

load_dotenv()

@dataclass
class Settings:
    use_azure: bool = True
    embed_deployment: str = os.getenv("AOAI_EMBED_DEPLOYMENT", "text-embedding-3-small")
    chat_deployment: str  = os.getenv("AOAI_CHAT_DEPLOYMENT",  "gpt-4o-mini")
    azure_endpoint: str   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str    = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str= os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    sqlite_path: str      = os.getenv("SQLITE_PATH", "./asset_examples.db")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./faiss_index_cosine.idx")

@dataclass(frozen=True)
class AssetClass:
    class_id: str         
    l1_code: str
    l1: str
    l2_code: str
    l2: str
    l3_code: str
    l3: str


class FaissStore:
    def __init__(self, settings: Settings, dim: int = 1536):
        self.settings = settings
        self.dim = dim
        self.path = self.settings.faiss_index_path
        self.index = faiss.IndexFlatIP(dim)  #cosine usng normalized vec
        self.next_vec_id = 0

    def open_db(self):
        conn = sqlite3.connect(self.settings.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # print("Opened DB at", S.sqlite_path)
        # print("Connnnnn", conn)
        return conn

    def load_or_init(self):
        if os.path.exists(self.path):
            self.index = faiss.read_index(self.path)
            with self.open_db() as conn:
                r = conn.execute("SELECT MAX(vec_id) FROM faiss_map").fetchone()
                self.next_vec_id = (r[0] + 1) if r and r[0] is not None else 0
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.next_vec_id = 0

    def save(self):
        faiss.write_index(self.index, self.path)

    def add_vectors(self, vecs: np.ndarray, row_ids: List[int]):
        n = vecs.shape[0]
        assert n == len(row_ids)
        self.index.add(vecs)
        with self.open_db() as conn:
            for i, rid in enumerate(row_ids):
                conn.execute(
                    "INSERT OR REPLACE INTO faiss_map (vec_id, row_id) VALUES (?,?)",
                    (self.next_vec_id + i, rid)
                )
            conn.commit()
        self.next_vec_id += n

    def search(self, qvec: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        qvec: shape (1, dim), already normalized.
        returns: (scores [1,k], vec_ids [1,k])
        """
        scores, ids = self.index.search(qvec, k)
        return scores, ids

    def vec_ids_to_row_ids(self, vec_ids: List[int]) -> List[int]:
        if not vec_ids:
            return []
        qmarks = ",".join(["?"] * len(vec_ids))
        with self.open_db() as conn:
            rows = conn.execute(
                f"SELECT vec_id, row_id FROM faiss_map WHERE vec_id IN ({qmarks})",
                vec_ids
            ).fetchall()
        mapping = {v: r for v, r in rows}
        return [mapping.get(v) for v in vec_ids]



class FeedbackRAG:
    def __init__(self, settings: Settings, vector_db_client: FaissStore):
        self.settings = settings
        self.vector_client = vector_db_client
        self.ALL_CLASSES: List[AssetClass] = []
        self.CLASS_BY_ID: Dict[str, AssetClass] = {}
        self.CLASS_ID_BY_TRIPLET: Dict[Tuple[str, str, str], str] = {}
        self.CLASS_ID_BY_CODE_TRIPLET: Dict[Tuple[str, str, str], str] = {}
        self.AOAI, self.OAI = self.get_clients()
        
    def load_all_classes_from_csv(self, path: str) -> None:
        """
        Loads the fixed asset registry CSV with columns:
        - Level 1 Code (Asset Class - OneERP)
        - Level 1 Description
        - Level 2 Code (Evaluation Group 4)
        - Asset Class 2 Description
        - Level 3 Code (Evaluation Group 5)
        - Asset Class 3 Description
        """
        self.ALL_CLASSES.clear(); self.CLASS_BY_ID.clear()
        self.CLASS_ID_BY_TRIPLET.clear(); self.CLASS_ID_BY_CODE_TRIPLET.clear()

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                l1_desc = html.unescape(row["Level 1 Description"]).strip()
                l2_desc = row["Asset Class 2 Description"].strip()
                l3_desc = row["Asset Class 3 Description"].strip()

                l1_code = row["Level 1 Code (Asset Class - OneERP)"].strip()
                l2_code = row["Level 2 Code (Evaluation Group 4)"].strip()
                l3_code = row["Level 3 Code (Evaluation Group 5)"].strip()

                cls = AssetClass(
                    class_id=l3_code,         
                    l1_code=l1_code, l1=l1_desc,
                    l2_code=l2_code, l2=l2_desc,
                    l3_code=l3_code, l3=l3_desc
                )
                self.ALL_CLASSES.append(cls)
                self.CLASS_BY_ID[cls.class_id] = cls
                #map by descriptions (mtches retrieved rows: r["l1"], r["l2"], r["l3"])
                self.CLASS_ID_BY_TRIPLET[(l1_desc, l2_desc, l3_desc)] = cls.class_id
                #map by code
                self.CLASS_ID_BY_CODE_TRIPLET[(l1_code, l2_code, l3_code)] = cls.class_id


    def candidate_classes_from_retrieved(self, retrieved_rows, max_candidates: int = 10) -> List[str]:
        seen = set()
        out: List[str] = []
        for r in retrieved_rows:
            triplet = (r["l1"].strip(), r["l2"].strip(), r["l3"].strip())
            cid = self.CLASS_ID_BY_TRIPLET.get(triplet)
            if cid and cid not in seen:
                seen.add(cid)
                out.append(cid)
            if len(out) >= max_candidates:
                break
        return out
    
    def get_clients(self):
        if self.settings.use_azure:
            # print("enddddddddd", self.settings.azure_endpoint)
            aoai = AzureOpenAI(
                api_key=self.settings.azure_api_key,
                api_version=self.settings.azure_api_version,
                azure_endpoint=self.settings.azure_endpoint
            )
            return aoai, None
        else:
            oai = OpenAI() 
            return None, oai
    
    def normalize_text(self, s: str) -> str:
        s = s.strip().lower()
        s = re.compile(r"\b((model|mdl|series|serial|sn|part|p\/n)\s*[:#]?\s*[A-Za-z0-9\-_.]+)\b", 
                       re.IGNORECASE).sub(" <model_token> ", s)   #####manufactr model
        s = re.compile(r"\b\d+\b").sub(" ", s)   #######numeric only
        s = re.compile(r"\s+").sub(" ", s).strip()    ####xtra space
        return s
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Returns L2-normalized embeddings (for cosine similarity with IndexFlatIP).
        """
        if self.AOAI:
            resp = self.AOAI.embeddings.create(
                model=self.settings.embed_deployment,
                input=texts
            )
            vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        else:
            resp = self.OAI.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = np.array([d.embedding for d in resp.data], dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms
    
    def open_db(self):
        conn = sqlite3.connect(self.settings.sqlite_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        # print("Opened DB at", S.sqlite_path)
        # print("Connnnnn", conn)
        return conn

    def init_db(self):
        SCHEMA_SQL = """
                    CREATE TABLE IF NOT EXISTS asset_examples (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        equip_description TEXT NOT NULL,
                        description1 TEXT,
                        manufacturer TEXT,
                        model_number TEXT,
                        l1 TEXT NOT NULL,
                        l2 TEXT NOT NULL,
                        l3 TEXT NOT NULL,
                        source TEXT DEFAULT 'seed',
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS faiss_map (
                        vec_id INTEGER PRIMARY KEY,  -- FAISS index id
                        row_id INTEGER NOT NULL,     -- SQLite row id
                        FOREIGN KEY(row_id) REFERENCES asset_examples(id)
                    );
                    """
        with self.open_db() as conn:
            for stmt in SCHEMA_SQL.strip().split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            conn.commit()
        
    def insert_example(self, equip_description:str, description: str, manufacturer: str, model_number: str, l1: str, l2: str, l3: str, source="seed") -> int:
        with self.open_db() as conn:
            now = datetime.datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO asset_examples (equip_description, description1, manufacturer, model_number, l1, l2, l3, source, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (equip_description, description, manufacturer, model_number,l1, l2, l3, source, now)
            )
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            conn.commit()
            return row_id

    def get_rows_by_ids(self, row_ids: List[int]) -> List[Dict[str, Any]]:
        if not row_ids:
            return []
        qmarks = ",".join(["?"] * len(row_ids))
        with self.open_db() as conn:
            rows = conn.execute(
                f"SELECT id, equip_description, description1, manufacturer, model_number, l1, l2, l3, source, created_at FROM asset_examples WHERE id IN ({qmarks})",
                row_ids
            ).fetchall()
        keys = ["id", "equip_description", "description1", "manufacturer", "model_number", "l1", "l2", "l3", "source", "created_at"]
        return [dict(zip(keys, r)) for r in rows]

    def get_row_by_rowid(self, row_id: int) -> Dict[str, Any]:
        with self.open_db() as conn:
            r = conn.execute(
                "SELECT id, equip_description, description1, manufacturer, model_number, l1, l2, l3, source, created_at FROM asset_examples WHERE id=?",
                (row_id,)
            ).fetchone()
        if not r:
            return {}
        keys = ["id", "equip_description", "description1", "manufacturer", "model_number", "l1", "l2", "l3", "source", "created_at"]
        return dict(zip(keys, r))

    def fetch_row_by_unique_key(self, unique_key: str):
        print("Fetching row by unique key:", unique_key)
        with self.open_db() as conn:
            print("Opened DB connection for fetch.")
            cur = conn.cursor()
            print("Executing query...")
            cur.execute(
                "SELECT id, equip_description, description1,l1, l2, l3 "
                "FROM asset_examples WHERE id = ?",
                (unique_key,)
            )
            row = cur.fetchone()
            print("Query executed.")
            # conn.close()
            print("Closed DB connection.")
            if row:
                print("Row found:", row)
                return {
                    "id": row[0],
                    "equip_description": row[1],
                    "description1": row[2],
                    "level1": row[3],
                    "level2": row[4],
                    "level3": row[5],
                }
            print("No row found for unique key.")
            return None
        
    def bulk_seed(self, examples: List[Dict[str, str]], batch: int = 512):
        """
        examples: [{description,l1,l2,l3}, ...]
        """
        self.vector_client.load_or_init()
        self.init_db()

        for i in range(0, len(examples), batch):
            chunk = examples[i:i+batch]
            descs = [self.normalize_text(e['equip_description']) for e in chunk]
            row_ids = [self.insert_example(d, e["description1"], e["manufacturer"], e["model_number"],e["l1"], e["l2"], e["l3"], source="seed") for d, e in zip(descs, chunk)]
            vecs = self.embed_texts(descs)
            self.vector_client.add_vectors(vecs, row_ids)
            self.vector_client.save()

    def build_user_prompt(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        examples_txt = "\n".join(
            [f"- desc: {r['equip_description']}\n  class: {r['l1']} > {r['l2']} > {r['l3']}" for r in retrieved]
        )
        return f"""
                Equipment description:
                {query}

                Retrieved examples (top-k):
                {examples_txt}

                Choose the most likely hierarchical class.
                """
    
    def call_llm_classify(self, query: str, retrieved: List[Dict[str, Any]]) -> Dict[str, Any]:

        # SYSTEM_PROMPT = """
        # You are an asset classification assistant.
        # You will receive an equipment description and a set of retrieved examples (with known class labels).

        # Rules:
        # - Predict a hierarchical class with three levels: level1, level2, level3.
        # - Consider domain terms, manufacturer/model patterns (they might be normalized), and functional hints.
        # - Prefer consensus among retrieved examples, but override if the description clearly indicates a different class.
        # - If uncertain, express lower confidence.

        # Return ONLY valid JSON:
        # {"level1": "...", "level2": "...", "level3": "...", "confidence": 0.0, "rationale": "..."}"""

        SYSTEM_PROMPT = """
        You are an asset classification assistant.

        You MUST classify into the existing taxonomy (2,389 classes).
        Each class is defined by (level1, level2, level3) and a unique class_id. You will be provided a small, allowed candidate list per query.

        STRICT RULES:
        - Choose EXACTLY one class_id from the allowed list provided in the user message.
        - NEVER invent or alter class names.
        - If uncertain, select the closest allowed candidate and lower the confidence.
        - Return only valid JSON that matches the schema.

        Return ONLY valid JSON of the form:
        {"class_id": "...", "confidence": 0.0, "rationale": "..."}
        """

        user_prompt = self.build_user_prompt(query, retrieved)
        if self.AOAI:
            resp = self.AOAI.chat.completions.create(
                model=self.settings.chat_deployment,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1
            )
            content = resp.choices[0].message.content
        else:
            resp = self.OAI.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1
            )
            content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except Exception:
            return {"level1":"", "level2":"", "level3":"", "confidence":0.0, "rationale":content}
    
    def retrieve(self, query: str, top_k: int = 8) -> Tuple[List[Dict[str, Any]], List[float]]:
        qn = self.normalize_text(query)
        qvec = self.embed_texts([qn])
        scores, ids = self.vector_client.search(qvec, k=top_k)
        vec_ids = [int(v) for v in ids[0] if v != -1]
        sims = [float(s) for s in scores[0][:len(vec_ids)]]
        row_ids = self.vector_client.vec_ids_to_row_ids(vec_ids)
        rows = self.get_rows_by_ids([rid for rid in row_ids if rid is not None])
        ordered = []
        for rid, s in zip(row_ids, sims):
            if rid is None:
                continue
            r = next((x for x in rows if x["id"] == rid), None)
            if r:
                ordered.append((r, s))
        return [o[0] for o in ordered], [o[1] for o in ordered]

    def classify(self, query: str, top_k: int = 8, llm_when_sim_below: float = 0.35) -> Dict[str, Any]:
        retrieved, sims = self.retrieve(query, top_k=top_k)

        result: Dict[str, Any] = {
            "query": query,
            "retrieved": [{"id": r["id"], "class": f"{r['l1']} > {r['l2']} > {r['l3']}", "sim": round(s, 4)} for r, s in zip(retrieved, sims)]
        }

        if sims and sims[0] >= 0.80:
            top = retrieved[0]
            result.update({
                "mode": "nearest_neighbor",
                "prediction": {"level1": top["l1"], "level2": top["l2"], "level3": top["l3"], "confidence": sims[0]},
                "rationale": "Top-1 retrieval very close; using nearest neighbor."
            })
            return result

        if not sims or sims[0] < llm_when_sim_below:
            pass
        llm_json = self.call_llm_classify(query, retrieved)
        result.update({
            "mode": "rag_llm",
            "prediction": {
                "level1": llm_json.get("level1",""),
                "level2": llm_json.get("level2",""),
                "level3": llm_json.get("level3",""),
                "confidence": float(llm_json.get("confidence", 0.0))
            },
            "rationale": llm_json.get("rationale","")
        })
        return result
    
    def submit_feedback( self, equip_description: str, description1: str, manufacturer: str, model_number: str, level1: str,
                        level2: str,level3: str,source="feedback") -> Dict[str, Any]:

        desc_n = self.normalize_text(equip_description)
        desc1 = self.normalize_text(description1)
        unique_key = f"{desc_n}|{level1}|{level2}|{level3}"
        print("Inserting feedback:", desc_n, level1, level2, level3)
        existing_row = self.fetch_row_by_unique_key(unique_key) 
        print("Existing row:", existing_row)
        if existing_row:
            return {
                "status": "duplicate",
                "row_id": existing_row["id"],
                "message": "Entry already exists. Skipping insert."
            }

        print("Inserting new entry.")
        row_id = self.insert_example( desc_n, description1,manufacturer,model_number,
                                     level1,level2,level3,# unique_key=unique_key,
                                     source=source)
        print("Inserted row ID:", row_id)

        vec = self.embed_texts([desc_n + " " + desc1])
        print("Embedding vector shape:", vec.shape)
        self.vector_client.add_vectors(vec, [row_id])
        print("Added vector to FAISS store.")
        self.vector_client.save()

        return {"status": "ok", "row_id": row_id}


    def test(self, path_to_dataset_csv):
        print("Loading FAISS & DB ...")
        self.vector_client.load_or_init()
        self.init_db()

        if self.vector_client.next_vec_id == 0:
            seed = []
            # db = pd.read_csv(r"C:\Users\MP946KB\WORKDIR\Embed-model\Hybrid-Approach\dataset\db.csv")
            db = pd.read_csv(path_to_dataset_csv)
            for i in range(len(db)):
                seed.append({
                    "equip_description": db.iloc[i]["Description"],
                    "description1": db.iloc[i]["Description.1"],
                    "manufacturer": db.iloc[i]["Manufacturer"],
                    "model_number": db.iloc[i]["Model number"],
                    "l1": db.iloc[i]["Level 1 Description"],
                    "l2": db.iloc[i]["Asset Class 2 Description"],
                    "l3": db.iloc[i]["Asset Class 3 Description"]
                })
            self.bulk_seed(seed)

        # Query
        q = "PIG LAUNCHER 4IN BALL VALVE"
        # res = classify(q, top_k=6)
        # print(json.dumps(res, indent=2))

        # Suppose the user corrects the class:
        # asset1,Asset2,Asset3,Description,Manufacturer,Model = "TS&T - Terminals","Terminal Equipment","Terminal Equipment Valves","L&T VALVES", "PM_VLV", "1-01-00-LP-1301-MIV-06"
        # fb = submit_feedback(q, "Mechanical", "Rotating Equipment", "Centrifugal Pump")
        # print(fb)