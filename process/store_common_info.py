import os
from dotenv import load_dotenv

load_dotenv()  # Load dari .env file

# Pastikan sudah ada di .env
# MONGO_CONNECTION_STRING=mongodb+srv://...
print("MONGO_CONNECTION_STRING:", os.environ.get("MONGO_CONNECTION_STRING", "NOT SET")[:30], "...")

import json

with open('../data/common_information.json', 'r', encoding='utf-8') as f:
    common_info_data = json.load(f)

print(f"Total dokumen: {len(common_info_data)}")
print("\nContoh dokumen pertama:")
print(json.dumps(common_info_data[0], indent=2))

from haystack import Document

documents = []
for item in common_info_data:
    # Content = gabungan question + answer untuk embedding yang lebih kontekstual
    content = f"Question: {item['question']}\nAnswer: {item['answer']}"
    
    doc = Document(
        content=content,
        meta={
            "topic": item["topic"],
            "question": item["question"],
        }
    )
    documents.append(doc)

print(f"Total Haystack Documents: {len(documents)}")
print("\nContoh Document pertama:")
print(documents[0])

from pymongo import MongoClient

client = MongoClient(os.environ['MONGO_CONNECTION_STRING'])
db = client['depato_store']

# Buat collection common_information jika belum ada
if 'common_information' not in db.list_collection_names():
    db.create_collection('common_information')
    print('Collection common_information berhasil dibuat.')
else:
    print('Collection common_information sudah ada.')

print(f"\nCollections di depato_store: {db.list_collection_names()}")

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore

# Document store khusus untuk common information
common_info_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="common_information",      # Collection terpisah dari produk
    vector_search_index="common_info_vector_index",  # Index vector baru
    full_text_search_index="common_info_search_index",
)

print("Document store berhasil dibuat.")
print(f"Collection: {common_info_store.collection_name}")

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

storing_pipeline = Pipeline()

storing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder()  # all-mpnet-base-v2, 768 dimensi
)
storing_pipeline.add_component(
    "writer",
    DocumentWriter(
        document_store=common_info_store,
        policy=DuplicatePolicy.OVERWRITE  # Aman untuk re-run
    )
)

storing_pipeline.connect("embedder", "writer")

print(storing_pipeline)

result = storing_pipeline.run({
    "embedder": {
        "documents": documents
    }
})

print(f"✅ Berhasil menyimpan {result['writer']['documents_written']} dokumen ke MongoDB Atlas")
print(f"   Collection: common_information")
print(f"   Database: depato_store")

# Cek jumlah dokumen di collection
collection = db['common_information']
count = collection.count_documents({})
print(f"Total dokumen di collection common_information: {count}")

# Tampilkan sample dokumen 
sample = collection.find_one({}, {"_id": 0, "content": 1, "meta": 1, "embedding": 1})
print("\nContoh dokumen tersimpan:")
print(f"  Topic: {sample.get('meta', {}).get('topic')}")
print(f"  Question: {sample.get('meta', {}).get('question')}")
print(f"  Content: {sample.get('content', '')[:100]}...")
print(f"  Embedding dimensions: {len(sample.get('embedding', []))}")

