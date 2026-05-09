from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder

from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
import os
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from typing import List
from getpass import getpass

import os
from dotenv import load_dotenv

load_dotenv()  # Load dari file .env di root project

# Pastikan MONGO_CONNECTION_STRING dan GROQ_API_KEY sudah diisi di file .env
print("MONGO:", os.environ.get("MONGO_CONNECTION_STRING", "NOT SET")[:40], "...")
print("GROQ :", os.environ.get("GROQ_API_KEY", "NOT SET")[:20], "...")

import os
from dotenv import load_dotenv

load_dotenv()  # Load dari file .env di root project

# Pastikan MONGO_CONNECTION_STRING dan GROQ_API_KEY sudah diisi di file .env
print("MONGO:", os.environ.get("MONGO_CONNECTION_STRING", "NOT SET")[:40], "...")
print("GROQ :", os.environ.get("GROQ_API_KEY", "NOT SET")[:20], "...")

document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

TEMPLATE = """"
You are a shop assiistant that helps users find the best products in a shopping mall.
You will be give a query and list of products. Your task is to generate a list of products that best match the query.

The output should be a list of products in the following format:

<summary_of_query>
<index>. <product_name> 
Price: <product_price>
Material: <product_material>
Category: <product_category>
Brand: <product_brand>
Recommendation: <product_recommendation>

From the format above, you should pay attention to the following:
1. <summary_of_query> should be a short summary of the query.
2. <index> should be a number starting from 1.
3. <product_name> should be the name of the product, this product name can be found from the product_name field.
4. <product_price> should be the price of the product, this product price can be found from the product_price field.
5. <product_material> should be the material of the product, this product material can be found from the product_material field.
6. <product_category> should be the category of the product, this product category can be found from the product_category field.
7. <product_brand> should be the brand of the product, this product brand can be found from the product_brand field.
8. <product_recommendation> should be the recommendation of the product, you should give a recommendation why this product is recommended, please pay attentation to the product_content field. 


You should only return the list of products that best match the query, do not return any other information.

The query is: {{query}}
the products are:
{% for product in documents %}
===========================================================
{{loop.index + 1}}. product_name: {{ product.meta.title }}
product_price: {{ product.meta.price }}
product_material: {{ product.meta.material }}
product_category: {{ product.meta.category }}
product_brand: {{ product.meta.brand }}
product_content: {{ product.content}}
{% endfor %}

===========================================================

Answer:

"""

pipeline = Pipeline()
pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5))
pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query", "documents"], required_variables=["query"]))
pipeline.add_component("generator", OpenAIChatGenerator(
    model="llama-3.3-70b-versatile",
    api_key=Secret.from_token(os.environ['GROQ_API_KEY']),
    api_base_url="https://api.groq.com/openai/v1"
))

pipeline.connect("embedder", "retriever")
pipeline.connect("retriever", "prompt_builder")
pipeline.connect("prompt_builder.prompt", "generator.messages")

query="I want to find an Outerwear that is not make me hot"
response = pipeline.run(
    {
        "embedder":{
            "text": query
        },
        "prompt_builder":{
            "query": query,
            "template": [ChatMessage.from_user(TEMPLATE)]
        }
    }
)

response['generator']['replies'][0].text


