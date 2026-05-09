import os
from getpass import getpass
os.environ["MONGO_CONNECTION_STRING"] = "mongodb+srv://alfiahzalfa:zalfa@alfiahzalfa.57hbmpr.mongodb.net/?appName=alfiahzalfa"

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever

document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store,top_k=10))

pipeline.connect("embedder", "retriever")

pipeline.run(
    {
        "embedder":{
            "text":"comfortable dress for going out with friends"
        }
    }
)

filters = {
    "field":"meta.category","operator":"==","value":"Dresses/Jumpsuits"
}

pipeline.run(
    {
        "embedder":{
            "text":"comfortable dress for going out with friends"
        },
        "retriever":{
            "filters": filters
        }
    }
)

filters = {
    "operator":"AND",
    "conditions":[
        {"field":"meta.category","operator":"==","value":"Dresses/Jumpsuits"},
        {"field":"meta.price", "operator":"<=", "value":50},
    ]
    
}

pipeline.run(
    {
        "embedder":{
            "text":"comfortable dress for going out with friends"
        },
        "retriever":{
            "filters": filters
        }
    }
)

filters = {
    "operator":"AND",
    "conditions":[
        {"field":"meta.category","operator":"==","value":"Dresses/Jumpsuits"},
        {"field":"meta.price", "operator":"<=", "value":50},
        {
            "operator":"OR",
            "conditions":[
                {"field":"meta.material","operator":"==","value":"Cotton"},
                {"field":"meta.material","operator":"==","value":"Polyester"}
            ]
        }
    ]
    
}

pipeline.run(
    {
        "embedder":{
            "text":"comfortable dress for going out with friends"
        },
        "retriever":{
            "filters": filters
        }
    }
)


