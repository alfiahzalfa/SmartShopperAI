import pandas as pd
df = pd.read_pickle('../data/datasets.pkl')

df.head()

from haystack import Document
documents = []
for index, row in df.iterrows():
    descriptions = row["description"].strip("[]").strip("''")

    doc = Document(
        content = f"{row['title']}\n {descriptions}",
        meta = {
            'asin': row['asin'],
            'title': row['title'],
            'brand': row['brand'],
            'price': row['price'],
            'gender': row['gender'],
            'material': row['material'],
            'category': row['category'],
        }
    )
    documents.append(doc)

documents[:5]

import os
from dotenv import load_dotenv

load_dotenv()  # Load dari file .env di root project

# Pastikan MONGO_CONNECTION_STRING sudah diisi di file .env
print("MONGO:", os.environ.get("MONGO_CONNECTION_STRING", "NOT SET")[:40], "...")

from pymongo import MongoClient

client = MongoClient(os.environ['MONGO_CONNECTION_STRING'])
db = client['depato_store']

if 'products' not in db.list_collection_names():
    db.create_collection('products')
    print('Collection products berhasil dibuat.')
else:
    print('Collection products sudah ada.')

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
pipeline_storing = Pipeline()

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

pipeline = Pipeline()
pipeline.add_component("embedder",SentenceTransformersDocumentEmbedder())
pipeline.add_component("writer",DocumentWriter(document_store=document_store,policy=DuplicatePolicy.OVERWRITE))

pipeline.connect("embedder","writer")

documents[:5]

pipeline.run({
    "embedder":{
        "documents":documents
    }
})

from pymongo import MongoClient
import os
client = MongoClient(os.environ['MONGO_CONNECTION_STRING'])
db = client.depato_store
material_collection = db.materials
category_collection = db.categories

materials = df['material'].unique().tolist()
categories = df['category'].unique().tolist()

documents_material= [ {"name":m} for m in materials]
documents_category = [ {"name":c} for c in categories]

material_collection.insert_many(documents_material)
category_collection.insert_many(documents_category)


