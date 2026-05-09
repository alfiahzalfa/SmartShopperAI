"""
SmartShopper Assistant — Streamlit App
AI Agent dengan Google ADK (FunctionTool) + Haystack RAG Pipelines + MongoDB Atlas

Arsitektur:
- Haystack  : RAG Pipelines (Embedder → MongoDB Retriever → LLM Generator)
- Google ADK: Agent + FunctionTool untuk routing otomatis antara:
    1. retrieve_and_generate_recommendation  → Product Recommendation Tool
    2. retrieve_common_information           → Common Information Tool

Fix ADK 1.33:
- Hapus 'break' setelah is_final_response() → mencegah GeneratorExit crash OpenTelemetry
- Event loop tidak di-set secara global (asyncio.set_event_loop dihindari)
- Pipeline registry hanya diisi satu kali, bukan setiap re-render
"""

import streamlit as st
import os
import re
import json
import asyncio
import uuid
import dotenv
from typing import List

from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from pymongo import MongoClient

# ── Google ADK Imports ────────────────────────────────────────────────────────
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types as genai_types

from template import METADATA_FILTER_TEMPLATE, COMMON_INFO_TEMPLATE

dotenv.load_dotenv()

# ── Global Pipeline Registry ──────────────────────────────────────────────────
# Diakses oleh Google ADK FunctionTool (module-level functions).
# Hanya diisi sekali dari Streamlit session_state, bukan tiap re-render.
_pipelines: dict = {}


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
    """Pipeline untuk mengekstrak metadata filter (category, material, price) dari query user."""

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


class RetrieveAndGenerateAnswerPipeline:
    """Pipeline RAG untuk rekomendasi produk: Embed → Retrieve → Generate."""

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

    def run(self, query: str, filter: dict = {}) -> str:
        messages = [
            ChatMessage.from_system(
                "You are a helpful shop assistant that gives product recommendations."
            ),
            ChatMessage.from_user(
                """
                Your task is to generate a list of products that best match the query.

                The output should be a list of products in the following format:

                <summary_of_query>
                <index>. <product_name>
                Price: <product_price>
                Material: <product_material>
                Category: <product_category>
                Brand: <product_brand>
                Recommendation: <product_recommendation>

                The query is: {{query}}
                {% if documents|length > 0 %}
                The products are:
                {% for product in documents %}
                ===========================================================
                {{loop.index}}. product_name: {{ product.meta.title }}
                product_price: {{ product.meta.price }}
                product_material: {{ product.meta.material }}
                product_category: {{ product.meta.category }}
                product_brand: {{ product.meta.brand }}
                product_content: {{ product.content}}
                {% endfor %}
                ===========================================================
                {% else %}
                There is no matching product.
                {% endif %}

                Answer:
                """
            ),
        ]
        res = self.pipeline.run(
            data={
                "embedder": {"text": query},
                "retriever": {"filters": filter},
                "prompt_builder": {"query": query, "template": messages},
            },
            include_outputs_from=["generator"],
        )
        return res["generator"]["replies"][0].text


class CommonInfoRAGPipeline:
    """
    Pipeline RAG untuk menjawab pertanyaan umum seputar e-commerce
    (pengiriman, pembelian, refund, return, dll.) dari data Common Information
    yang tersimpan di MongoDB Atlas collection 'common_information'.
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


# ── Google ADK FunctionTools ──────────────────────────────────────────────────
# Fungsi di bawah ini di-wrap oleh Google ADK sebagai FunctionTool secara otomatis
# saat dimasukkan ke dalam list tools=[] pada Agent.
# Docstring berfungsi sebagai deskripsi tool untuk LLM (KRITIS untuk routing).

def retrieve_and_generate_recommendation(query: str) -> str:
    """
    Retrieves products from the catalog and generates personalized recommendations.

    Use this tool when the user asks about specific products, wants product
    recommendations, or inquires about items such as dresses, tops, shoes,
    accessories, specific price ranges, materials, brands, or categories.

    Args:
        query: The user's product-related query.
               Example: "I want a cotton dress under $50" or "show me leather accessories".

    Returns:
        A formatted list of recommended products with name, price, material,
        category, brand, and recommendation reason.
    """
    metadata_filter_pipeline = _pipelines.get("metadata_filter")
    rag_pipeline = _pipelines.get("rag")

    if not metadata_filter_pipeline or not rag_pipeline:
        return "Error: RAG pipeline belum diinisialisasi."

    # Step 1: Extract metadata filters (category, material, price) from query
    filter_result = metadata_filter_pipeline.run(query)
    filters = {}
    try:
        json_match = re.search(r"```json\n(.*?)\n```", filter_result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
            # Hanya gunakan filter jika bukan dict kosong
            if parsed:
                filters = parsed
    except Exception:
        filters = {}

    # Step 2: Retrieve matching products and generate recommendation
    return rag_pipeline.run(query, filters)


def retrieve_common_information(query: str) -> str:
    """
    Answers general e-commerce questions about shopping processes and policies.

    Use this tool when the user asks about: shipping and delivery time, order tracking,
    how to place or cancel an order, refund policy, how to request a refund,
    how to return an item, return eligibility, payment methods, account management,
    loyalty programs, gift cards, or any general e-commerce inquiry.

    Do NOT use this tool for product recommendations or product searches.

    Args:
        query: The user's general e-commerce question.
               Example: "How do I return an item?" or "What is your refund policy?".

    Returns:
        A detailed, friendly answer based on e-commerce policy documents.
    """
    common_info_pipeline = _pipelines.get("common_info")

    if not common_info_pipeline:
        return "Error: Common info pipeline belum diinisialisasi."

    return common_info_pipeline.run(query)


# ── Google ADK Agent Runner Helpers ───────────────────────────────────────────

async def _call_agent_async(
    runner: Runner, user_id: str, session_id: str, query: str
) -> str:
    """
    Jalankan ADK agent secara async dan return final response.

    PENTING (ADK 1.33 fix):
    - JANGAN gunakan 'break' setelah is_final_response().
    - 'break' memicu GeneratorExit yang crash di OpenTelemetry context vars.
    - Solusi: iterasi semua events sampai habis, simpan yang is_final_response=True.
    """
    content = genai_types.Content(role="user", parts=[genai_types.Part(text=query)])
    final_response = ""

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content,
    ):
        # Kumpulkan semua final response (ambil yang terakhir)
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text
        # TIDAK ada 'break' di sini — biarkan generator selesai natural

    return final_response


def _run_agent_sync(runner: Runner, user_id: str, session_id: str, query: str) -> str:
    """
    Wrapper sinkron untuk menjalankan ADK agent di dalam Streamlit.

    Membuat event loop baru yang terisolasi agar tidak bentrok dengan
    thread Streamlit. TIDAK memanggil asyncio.set_event_loop() secara global
    untuk menghindari side-effect pada thread lain.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _call_agent_async(runner, user_id, session_id, query)
        )
    except Exception as e:
        return f"⚠️ Terjadi error saat memproses pertanyaan kamu: {str(e)}"
    finally:
        loop.close()


def response_handler(query: str) -> str:
    """Handler utama yang dipanggil Streamlit saat user mengirim pesan."""
    response_text = _run_agent_sync(
        runner=st.session_state.adk_runner,
        user_id="smartshopper_user",
        session_id=st.session_state.adk_session_id,
        query=query,
    )

    # Fallback jika response kosong
    if not response_text or not response_text.strip():
        response_text = "Maaf, saya tidak mendapatkan respons. Silakan coba lagi."

    st.session_state.display_messages.append({"role": "user", "content": query})
    st.session_state.display_messages.append({"role": "assistant", "content": response_text})
    return response_text


# ── Helper: init ADK session (sync wrapper) ───────────────────────────────────

def _create_session_sync(session_service: InMemorySessionService, session_id: str) -> None:
    """Buat ADK session secara sinkron tanpa mempengaruhi global event loop."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            session_service.create_session(
                app_name="smartshopper",
                user_id="smartshopper_user",
                session_id=session_id,
            )
        )
    finally:
        loop.close()


# ── Streamlit App ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    st.set_page_config(
        page_title="SmartShopper Assistant",
        page_icon="🛍️",
        layout="centered",
    )
    st.title("🛍️ SmartShopper Assistant")
    st.caption("Powered by Google ADK + Haystack + MongoDB Atlas")

    # ── Init display messages ─────────────────────────────────────────────────
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    # ── Init MongoDB Document Stores ──────────────────────────────────────────
    if "document_store" not in st.session_state:
        st.session_state.document_store = MongoDBAtlasDocumentStore(
            database_name="depato_store",
            collection_name="products",
            vector_search_index="vector_index",
            full_text_search_index="search_index",
        )

    if "common_info_document_store" not in st.session_state:
        st.session_state.common_info_document_store = MongoDBAtlasDocumentStore(
            database_name="depato_store",
            collection_name="common_information",
            vector_search_index="common_info_vector_index",
            full_text_search_index="common_info_search_index",
        )

    # ── Init Haystack RAG Pipelines ───────────────────────────────────────────
    if "metadata_filter_pipeline" not in st.session_state:
        with st.spinner("Initializing metadata filter pipeline..."):
            st.session_state.metadata_filter_pipeline = MetaDataFilterPipeline()

    if "retrieve_and_generate_pipeline" not in st.session_state:
        with st.spinner("Initializing product retrieval pipeline..."):
            st.session_state.retrieve_and_generate_pipeline = RetrieveAndGenerateAnswerPipeline(
                document_store=st.session_state.document_store
            )

    if "common_info_pipeline" not in st.session_state:
        with st.spinner("Initializing common info pipeline..."):
            st.session_state.common_info_pipeline = CommonInfoRAGPipeline(
                document_store=st.session_state.common_info_document_store
            )

    # ── Register pipelines ke global registry (hanya jika belum terisi) ──────
    # Guard ini penting: cegah overwrite tiap re-render Streamlit
    if not _pipelines:
        _pipelines["metadata_filter"] = st.session_state.metadata_filter_pipeline
        _pipelines["rag"] = st.session_state.retrieve_and_generate_pipeline
        _pipelines["common_info"] = st.session_state.common_info_pipeline

    # ── Init Google ADK Agent ─────────────────────────────────────────────────
    if "adk_agent" not in st.session_state:
        st.session_state.adk_agent = Agent(
            name="smartshopper_agent",
            model=LiteLlm(model="groq/llama-3.3-70b-versatile"),
            # Passing plain Python functions → Google ADK otomatis wrap sebagai FunctionTool
            tools=[
                retrieve_and_generate_recommendation,
                retrieve_common_information,
            ],
            instruction="""
            You are a helpful SmartShopper assistant for an e-commerce platform.

            You have access to two tools:
            1. retrieve_and_generate_recommendation — Use ONLY for product-related questions.
               Examples: "I want a cotton dress", "show me tops under $50", "recommend shoes for women".
            2. retrieve_common_information — Use ONLY for general e-commerce questions.
               Examples: "how do I return an item?", "what is your refund policy?",
               "how long does shipping take?", "how do I place an order?", "what payment methods are accepted?".

            ROUTING RULES (STRICTLY FOLLOW THIS):
            - Product query or product recommendation request → use retrieve_and_generate_recommendation.
            - Shipping, refund, return, purchasing process, payment, account question → use retrieve_common_information.
            - Simple greeting or completely unrelated to shopping → respond directly, no tools needed.

            Always provide a clear, complete, and friendly final answer based on the tool result.
            If the question is outside the shopping context, politely inform the user.
            """,
        )

    # ── Init Google ADK Session Service + Session ─────────────────────────────
    if "adk_session_service" not in st.session_state:
        session_service = InMemorySessionService()
        session_id = str(uuid.uuid4())

        # Buat session baru menggunakan isolated event loop (tidak polute global loop)
        _create_session_sync(session_service, session_id)

        st.session_state.adk_session_service = session_service
        st.session_state.adk_session_id = session_id

    # ── Init Google ADK Runner ────────────────────────────────────────────────
    if "adk_runner" not in st.session_state:
        st.session_state.adk_runner = Runner(
            agent=st.session_state.adk_agent,
            app_name="smartshopper",
            session_service=st.session_state.adk_session_service,
        )

    # ── Render Chat History ───────────────────────────────────────────────────
    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── Chat Input ────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Hello, what can I help you with today?"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            response = response_handler(prompt)

        with st.chat_message("assistant"):
            st.markdown(response)
