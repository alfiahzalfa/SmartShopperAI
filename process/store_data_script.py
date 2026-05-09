"""
TAHAP 4 — Store Data Produk ke MongoDB Atlas
Jalankan: .\venv\Scripts\python.exe process\store_data_script.py
"""

import os
import sys
from dotenv import load_dotenv

# Load .env dari root project
load_dotenv()

MONGO_URI = os.environ.get("MONGO_CONNECTION_STRING", "")
if not MONGO_URI:
    print("❌ ERROR: MONGO_CONNECTION_STRING tidak ditemukan di .env!")
    sys.exit(1)

print(f"✅ MONGO: {MONGO_URI[:40]}...")

# ── 1. Load Dataset ────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset dari data/datasets.pkl ...")
import pandas as pd
df = pd.read_pickle("data/datasets.pkl")
print(f"      Total rows: {len(df)}")

# ── 2. Konversi ke Haystack Documents ─────────────────────────────────────────
print("\n[2/5] Membuat Haystack Documents ...")
from haystack import Document

documents = []
for index, row in df.iterrows():
    descriptions = row["description"].strip("[]").strip("''")
    doc = Document(
        content=f"{row['title']}\n {descriptions}",
        meta={
            "asin":     row["asin"],
            "title":    row["title"],
            "brand":    row["brand"],
            "price":    row["price"],
            "gender":   row["gender"],
            "material": row["material"],
            "category": row["category"],
        },
    )
    documents.append(doc)

print(f"      Total documents: {len(documents)}")

# ── 3. Setup MongoDB Collection ────────────────────────────────────────────────
print("\n[3/5] Menyiapkan MongoDB collection ...")
from pymongo import MongoClient

client = MongoClient(MONGO_URI)
db = client["depato_store"]

if "products" not in db.list_collection_names():
    db.create_collection("products")
    print("      Collection 'products' berhasil dibuat.")
else:
    print("      Collection 'products' sudah ada.")

# ── 4. Simpan Materials & Categories ──────────────────────────────────────────
print("\n[4/5] Menyimpan materials dan categories ...")
material_collection = db.materials
category_collection = db.categories

materials  = df["material"].unique().tolist()
categories = df["category"].unique().tolist()

# Hapus dulu biar tidak duplikat
material_collection.drop()
category_collection.drop()

material_collection.insert_many([{"name": m} for m in materials])
category_collection.insert_many([{"name": c} for c in categories])

print(f"      Materials saved : {len(materials)}")
print(f"      Categories saved: {len(categories)}")

# ── 5. Embedding + Upload ke MongoDB ──────────────────────────────────────────
print("\n[5/5] Embedding dan upload produk ke MongoDB Atlas ...")
print("      ⚠️  Proses ini bisa memakan waktu 5-15 menit ...\n")

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder())
pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
)
pipeline.connect("embedder", "writer")

result = pipeline.run({"embedder": {"documents": documents}})

print(f"\n✅ Selesai! {result['writer']['documents_written']} produk berhasil disimpan ke MongoDB Atlas.")
print("   Database  : depato_store")
print("   Collection: products")
print("\n👉 Lanjut ke Tahap 5: jalankan store_common_info_script.py")
