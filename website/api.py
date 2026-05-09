"""
SmartShopper Assistant — FastAPI Endpoint
Menyajikan AI Agent (Product Recommendation + Common Information) sebagai REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import re
import json
import os
import dotenv

dotenv.load_dotenv()

from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from pymongo import MongoClient
from typing import List

from template import METADATA_FILTER_TEMPLATE, COMMON_INFO_TEMPLATE

# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SmartShopper Assistant API",
    description="AI Agent untuk rekomendasi produk dan informasi umum e-commerce menggunakan Google ADK + Haystack + MongoDB Atlas",
    version="2.0.0",
)

# ── Request / Response Models ─────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

class RecommendResponse(BaseModel):
    query: str
    recommendation: str

class CommonInfoRequest(BaseModel):
    query: str

class CommonInfoResponse(BaseModel):
    query: str
    answer: str

# ── Haystack Components ───────────────────────────────────────────────────────

@component
class GetMaterials:
    def __init__(self):
        client = MongoClient(os.environ["MONGO_CONNECTION_STRING"])
        self.collection = client.depato_store.materials

    @component.output_types(materials=List[str])
    def run(self):
        return {"materials": [doc["name"] for doc in self.collection.find()]}


@component
class GetCategories:
    def __init__(self):
        client = MongoClient(os.environ["MONGO_CONNECTION_STRING"])
        self.collection = client.depato_store.categories

    @component.output_types(categories=List[str])
    def run(self):
        return {"categories": [doc["name"] for doc in self.collection.find()]}


# ── Haystack RAG Pipelines ────────────────────────────────────────────────────

class MetaDataFilterPipeline:
    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.add_component("materials", GetMaterials())
        self.pipeline.add_component("categories", GetCategories())
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                variables=["input", "materials", "categories"],
                required_variables=["input", "materials", "categories"],
            ),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="llama-3.3-70b-versatile",
                api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
                api_base_url="https://api.groq.com/openai/v1",
            ),
        )
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str) -> str:
        template = [ChatMessage.from_user(METADATA_FILTER_TEMPLATE)]
        res = self.pipeline.run(
            data={"prompt_builder": {"input": query, "template": template}},
        )
        return res["generator"]["replies"][0].text


class RetrieveAndGeneratePipeline:
    def __init__(self, document_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component(
            "retriever",
            MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=10),
        )
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                variables=["query", "documents"],
                required_variables=["query", "documents"],
            ),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="llama-3.3-70b-versatile",
                api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
                api_base_url="https://api.groq.com/openai/v1",
            ),
        )
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str, filters: dict = {}) -> str:
        messages = [
            ChatMessage.from_system(
                "You are a helpful shop assistant that gives product recommendations."
            ),
            ChatMessage.from_user(
                """
                Give a list of products that best match the query.

                Format each product as:
                <index>. <product_name>
                Price: <price>
                Material: <material>
                Category: <category>
                Brand: <brand>
                Recommendation: <why recommended>

                Query: {{query}}
                {% if documents|length > 0 %}
                Products:
                {% for p in documents %}
                ---
                {{loop.index}}. {{p.meta.title}} | ${{p.meta.price}} | {{p.meta.material}} | {{p.meta.category}}
                {{p.content}}
                {% endfor %}
                {% else %}
                No matching products found.
                {% endif %}

                Answer:
                """
            ),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "retriever": {"filters": filters},
                "prompt_builder": {"query": query, "template": messages},
            },
            include_outputs_from=["generator"],
        )
        return res["generator"]["replies"][0].text


class CommonInfoRAGPipeline:
    """
    Pipeline RAG untuk menjawab pertanyaan umum e-commerce
    dari data Common Information di MongoDB Atlas.
    """
    def __init__(self, document_store):
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component(
            "retriever",
            MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5),
        )
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                variables=["query", "documents"],
                required_variables=["query", "documents"],
            ),
        )
        self.pipeline.add_component(
            "generator",
            OpenAIChatGenerator(
                model="llama-3.3-70b-versatile",
                api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
                api_base_url="https://api.groq.com/openai/v1",
            ),
        )
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str) -> str:
        messages = [
            ChatMessage.from_system(
                "You are a helpful e-commerce customer support assistant. "
                "Answer questions about shipping, purchasing, refunds, and returns based on the provided documents."
            ),
            ChatMessage.from_user(COMMON_INFO_TEMPLATE),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "prompt_builder": {"query": query, "template": messages},
            },
            include_outputs_from=["generator"],
        )
        return res["generator"]["replies"][0].text


# ── Singleton Pipelines (inisialisasi saat startup) ───────────────────────────

product_document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

common_info_document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="common_information",
    vector_search_index="common_info_vector_index",
    full_text_search_index="common_info_search_index",
)

metadata_filter_pipeline = MetaDataFilterPipeline()
rag_pipeline = RetrieveAndGeneratePipeline(product_document_store)
common_info_pipeline = CommonInfoRAGPipeline(common_info_document_store)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Cek status API."""
    return {"status": "healthy", "service": "SmartShopper API v2.0"}


@app.post("/recommend", response_model=RecommendResponse)
def get_recommendation(request: RecommendRequest):
    """
    Product Recommendation Tool:
    Terima query produk → filter metadata → retrieve → generate rekomendasi.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    # Extract metadata filters
    raw_filter = metadata_filter_pipeline.run(request.query)
    filters = {}
    try:
        match = re.search(r"```json\n(.*?)\n```", raw_filter, re.DOTALL)
        if match:
            filters = json.loads(match.group(1))
    except Exception:
        filters = {}

    # Retrieve + generate
    recommendation = rag_pipeline.run(request.query, filters)

    return RecommendResponse(
        query=request.query,
        recommendation=recommendation,
    )


@app.post("/common-info", response_model=CommonInfoResponse)
def get_common_info(request: CommonInfoRequest):
    """
    Common Information Tool:
    Terima pertanyaan umum e-commerce → retrieve dari FAQ → generate jawaban.
    Contoh: shipping, refund, return, payment, account.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong.")

    answer = common_info_pipeline.run(request.query)

    return CommonInfoResponse(
        query=request.query,
        answer=answer,
    )


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)