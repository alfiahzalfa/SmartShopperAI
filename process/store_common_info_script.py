"""
TAHAP 5 — Store Common Information ke MongoDB Atlas
Jalankan: .\venv\Scripts\python.exe process\store_common_info_script.py
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load .env dari root project
load_dotenv()

MONGO_URI = os.environ.get("MONGO_CONNECTION_STRING", "")
if not MONGO_URI:
    print("❌ ERROR: MONGO_CONNECTION_STRING tidak ditemukan di .env!")
    sys.exit(1)

print(f"✅ MONGO: {MONGO_URI[:40]}...")

# ── 1. Load Common Information JSON ───────────────────────────────────────────
print("\n[1/4] Loading common_information.json ...")
with open("data/common_information.json", "r", encoding="utf-8") as f:
    common_info_data = json.load(f)

print(f"      Total entries: {len(common_info_data)}")
print(f"      Contoh pertama: [{common_info_data[0]['topic']}] {common_info_data[0]['question']}")

# ── 2. Konversi ke Haystack Documents ─────────────────────────────────────────
print("\n[2/4] Membuat Haystack Documents ...")
from haystack import Document

documents = []
for item in common_info_data:
    content = f"Question: {item['question']}\nAnswer: {item['answer']}"
    doc = Document(
        content=content,
        meta={
            "topic":    item["topic"],
            "question": item["question"],
        },
    )
    documents.append(doc)

print(f"      Total documents: {len(documents)}")

# ── 3. Setup MongoDB Collection ────────────────────────────────────────────────
print("\n[3/4] Menyiapkan MongoDB collection ...")
from pymongo import MongoClient

client = MongoClient(MONGO_URI)
db = client["depato_store"]

if "common_information" not in db.list_collection_names():
    db.create_collection("common_information")
    print("      Collection 'common_information' berhasil dibuat.")
else:
    print("      Collection 'common_information' sudah ada.")

print(f"      Collections saat ini: {db.list_collection_names()}")

# ── 4. Embedding + Upload ke MongoDB ──────────────────────────────────────────
print("\n[4/4] Embedding dan upload common information ke MongoDB Atlas ...")

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

common_info_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="common_information",
    vector_search_index="common_info_vector_index",
    full_text_search_index="common_info_search_index",
)

storing_pipeline = Pipeline()
storing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
storing_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=common_info_store, policy=DuplicatePolicy.OVERWRITE),
)
storing_pipeline.connect("embedder", "writer")

result = storing_pipeline.run({"embedder": {"documents": documents}})

print(f"\n✅ Selesai! {result['writer']['documents_written']} dokumen FAQ berhasil disimpan ke MongoDB Atlas.")
print("   Database  : depato_store")
print("   Collection: common_information")

# ── Verifikasi ─────────────────────────────────────────────────────────────────
collection = db["common_information"]
count = collection.count_documents({})
print(f"\n📊 Verifikasi: {count} dokumen ada di collection 'common_information'")

print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅ Data sudah tersimpan di MongoDB!                         ║
║                                                              ║
║  LANGKAH SELANJUTNYA:                                        ║
║  1. Buka cloud.mongodb.com                                   ║
║  2. Buat Vector Search Index untuk collection 'products'     ║
║     - Index name: vector_index                               ║
║     - dimensions: 768, similarity: cosine                    ║
║  3. Buat Vector Search Index untuk 'common_information'      ║
║     - Index name: common_info_vector_index                   ║
║     - dimensions: 768, similarity: cosine                    ║
║  4. Tunggu index status = Active                             ║
║  5. Jalankan: streamlit run website\\website.py               ║
╚══════════════════════════════════════════════════════════════╝
""")
