from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack.utils import Secret
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from pymongo import MongoClient
from typing import List,Annotated
from getpass import getpass
import re
import json
import os

import logging
from haystack import tracing
from haystack.tracing.logging_tracer import LoggingTracer

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.DEBUG)

tracing.tracer.is_content_tracing_enabled = True # to enable tracing/logging content (inputs/outputs)
tracing.enable_tracing(LoggingTracer(tags_color_strings={"haystack.component.input": "\x1b[1;31m", "haystack.component.name": "\x1b[1;34m"}))

os.environ["GROQ_API_KEY"] = getpass("Masukkan Groq API Key Anda: ")

os.environ["MONGO_CONNECTION_STRING"] = getpass("Masukkan MongoDB Connection String Anda: ")

chat_message_store = InMemoryChatMessageStore()
document_store = MongoDBAtlasDocumentStore(
    database_name="depato_store",
    collection_name="products",
    vector_search_index="vector_index",
    full_text_search_index="search_index",
)

class ParaphraserPipeline:
    def __init__(self,chat_message_store):
        self.memory_retriever = ChatMessageRetriever(chat_message_store)
        # self.memory_retriever = memory_retriever
        # self.memory_writer = memory_writer
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder",ChatPromptBuilder(variables=["query","memories"],required_variables=["query", "memories"],))
        self.pipeline.add_component("generator", OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"))
        self.pipeline.add_component("memory_retriever", self.memory_retriever)

        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")
    
    def run(self, query):
        messages = [
            ChatMessage.from_system(
                "You are a helpful assistant that paraphrases user queries based on previous conversations."
            ),
            ChatMessage.from_user(
                """
                Please paraphrase the following query based on the conversation history provided below. If the conversation history is empty, please return the query as is.
                history:
                {% for memory in memories %}
                    {{memory.text}}
                {% endfor %}
                query: {{query}}
                answer:
                """
            )
        ]

        res = self.pipeline.run(
            data = {
                "prompt_builder":{
                    "query": query,
                    "template": messages
                },
                "memory_retriever": {           # ← ini yang masih kurang
                    "chat_history_id": "default"
                }
                # "joiner":{
                #     "values":  [ChatMessage.from_user(query)]
                # }
            },
            include_outputs_from=["generator"]
        )
        print("Pipeline Input", query)
        return res["generator"]["replies"][0].text

paraprahser_pipeline = ParaphraserPipeline(chat_message_store=chat_message_store)

class ChatHistoryPipeline:
    def __init__(self, chat_message_store):
        self.chat_message_store = chat_message_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("memory_retriever", ChatMessageRetriever(chat_message_store))
        self.pipeline.add_component("prompt_builder", PromptBuilder(variables=["memories"], required_variables=["memories"], template="""
        Previous Conversations history:
        {% for memory in memories %}
            {{memory.text}}
        {% endfor %}
        """)
        )
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")

    def run(self):
        res = self.pipeline.run(
            data = {
                "memory_retriever": {
                    "chat_history_id": "default"
                }
            },
            include_outputs_from=["prompt_builder"]
        )
        return res["prompt_builder"]["prompt"]

        

chat_history_pipeline = ChatHistoryPipeline(chat_message_store=chat_message_store)

class MongoDBAtlas:
    def __init__(self, mongo_connection_string:str):
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client.depato_store
        self.material_collection = self.db.materials
        self.category_collection = self.db.categories

    def get_materials(self):
        return [doc['name'] for doc in self.material_collection.find()]

    def get_categories(self):
        return [doc['name'] for doc in self.category_collection.find()]

@component
class GetMaterials:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(materials=List[str])
    def run(self):
        materials = self.db.get_materials()
        return {"materials": materials}

@component
class GetCategories:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(categories=List[str])
    def run(self):
        categories = self.db.get_categories()
        return {"categories": categories}

METADATA_FILTER_TEMPLATE = """
You are a json generator that have a job to generate json based on the input.
The return json should be in the format:
```json
{
    "operator": "AND",
    "conditions":[
        {"field": "meta.category", "operator":"==", "value": <category>},
        {"field": "meta.material", "operator":"==", "value": <material>},
        {"filed": "meta.gender", "operator":"==", "value" : <male|female|unisex>},
        {"field": "meta.price", "operator":<"<="|">="|"==">, "value": <price>}
    ]
}
```
The json key above can be omiitted if the value is not provided in the input, so please make sure to only return the keys that are provided in the input.

For the material and category, you can only use the material and category that are provided below:
Materials: [ {% for material in materials %} {{ material }} {% if not loop.last %}, {% endif %} {% endfor %} ]

Categories: [ {% for category in categories %} {{ category }} {% if not loop.last %}, {% endif %} {% endfor %} ]

if the input does not contain any of the keys above, you should return an empty json object like this:
```json
{}
```
Sometimes the material and category can be negated, so you should also handle that by using the operator "!=" for material and category. 

Sometimes the material and category is not explicitly mentioned, you should analyze which material and category is the most suitable based on the input, and return the json with the material and category that you think is the most suitable.

Nestede conditions are allowed, for nested conditions, you can use "OR" and "AND" as the operator, and the conditions should be in the "conditions" array.

if user said the price around some value, please find the price between those value -10 and value +10.

The example of the result are expected to be like this:

1. Input: "can you give me a adress with cotton material?"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.material", "operator": "==", "value": "Cotton"},
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"}
    ]
}
```

2. Input: "Give me Shirt that is not made of cotton and has a price less than $100"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Tops"},
        {"field": "meta.material", "operator": "!=", "value": "Cotton"},
        {"field": "meta.price", "operator": "<=", "value": 100}
    ]
}
3. Input: "I want a dress that is not hot and has a price greater than $50"
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"},
        {"field": "meta.price", "operator": ">=", "value": 50},
        {
            "operator": "OR",
            "conditions": [
                {"field": "meta.material", "operator": "==", "value": "Cotton"},
                {"field": "meta.material", "operator": "==", "value": "Polyester"}
            ]
        }
    ]
}

4. Input i want tops that have price between $20 and $50
output:
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Tops"},
        {
            "operator": "AND",
            "conditions":[
                {"field": "meta.price", "operator": ">=", "value": 20},
                {"field": "meta.price", "operator": "<=", "value": 50}
            ]
        }
    ]
}
```
5. Input: I want the dress price around $50
output: 
```json
{
    "operator": "AND",
    "conditions": [
        {"field": "meta.category", "operator": "==", "value": "Dresses/Jumpsuits"},
        {
            "operator": "AND",
            "conditions":[
                {"field": "meta.price", "operator": ">=", "value": 40},
                {"field": "meta.price", "operator": "<=", "value": 60}
            ]
        }
    ]
}
```
6. Input: {{input}}
output:

```

"""

class MetaDataFilterPipeline:
    def __init__(self, get_materials, get_categories, template):
        self.get_materials = get_materials
        self.get_categories = get_categories
        self.template = template

        self.pipeline = Pipeline()
        self.pipeline.add_component("materials", GetMaterials())
        self.pipeline.add_component("categories", GetCategories())
        self.pipeline.add_component(
            "prompt_builder",
            ChatPromptBuilder(
                variables=["input", "materials", "categories"],
                required_variables=["input", "materials", "categories"],
            )
        )
        self.pipeline.add_component("generator", OpenAIChatGenerator(
            model="llama-3.3-70b-versatile",
            api_key=Secret.from_token(os.environ['GROQ_API_KEY']),
            api_base_url="https://api.groq.com/openai/v1"
        ))
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder.prompt","generator.messages")

    def run(self, query: str):
        template = [ChatMessage.from_user(self.template)]
        res = self.pipeline.run(
            data={
                "prompt_builder": {
                    "input": query,
                    "template": template,
                },
            },
        )
        return res["generator"]["replies"][0].text

class RetrieveAndGenerateAnswerPipeline:
    def __init__(self, chat_message_store, document_store):
        self.chat_message_store = chat_message_store
        self.document_store = document_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store,top_k=10))
        self.pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query","documents"],required_variables=["query", "documents"]))
        self.pipeline.add_component("generator", OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"))
        # self.pipeline.add_component("chat_message_writer", ChatMessageWriter(chat_message_store))
        # self.pipeline.add_component("joiner", ListJoiner(List[ChatMessage]))
        
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        # self.pipeline.connect("generator.replies", "joiner")
        # self.pipeline.connect("joiner", "chat_message_writer")

    def run(self, query: str, filter: dict = {}):
        messages = [
            ChatMessage.from_system(
                "You are a helpful shop assistant that will give products recommendation based on user query and metadata filtering. "
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

                if there is no matching product below, please say so.

                The query is: {{query}}
                {% if documents|length > 0 %}
                the products are:
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
            )
        ]
        res = self.pipeline.run(
            data={
                "embedder":{
                    "text": query,
                },
                "retriever":{
                    "filters":filter
                },
               "prompt_builder":{
                   "query": query,
                   "template": messages
               }

            },
            include_outputs_from=["generator", "prompt_builder"]
        )
        print(res["prompt_builder"]["prompt"])
        return res["generator"]["replies"][0].text

retrieve_and_generate_pipeline = RetrieveAndGenerateAnswerPipeline(chat_message_store=chat_message_store, document_store=document_store)
metadata_filter_pipeline = MetaDataFilterPipeline(
    get_materials=GetMaterials(),
    get_categories=GetCategories(),
    template=METADATA_FILTER_TEMPLATE
)

def retrieve_and_generate(query: Annotated[str, "User query"]):
    """
    This tool retrieves products based on user query and generates an answer.
    """
    pharaprased_query = paraprahser_pipeline.run(query)
    result = metadata_filter_pipeline.run(pharaprased_query)
    data = {}
    try:
        json_match = re.search(r'```json\n(.*?)\n```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
        else:
            logging.error("No JSON found in the result.")
            data = {}
    except Exception as e:
        logging.error(f"Error parsing JSON from result: {e}")
        data = {}
    

    return retrieve_and_generate_pipeline.run(pharaprased_query,data)

retrieve_and_generate_tool = Tool(
    name="recommend",
    description="Use this tool to retrieve and recommend products based on user query.",
    function=retrieve_and_generate,
    parameters= {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user query to retrieve products and generate an answer."
            }
        },
        "required": ["query"]
    }
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(
        model="meta-llama/llama-4-scout-17b-16e-instruct",  # ← model khusus tool use
        api_key=Secret.from_token(os.environ["GROQ_API_KEY"]),
        api_base_url="https://api.groq.com/openai/v1"
    ),
    tools=[retrieve_and_generate_tool],
    system_prompt="""
    You are a helpful shop assistant that provides product recommendations.

    STRICT RULES:
    - For greetings or non-product questions: respond with text ONLY, do NOT call any tool.
    - For product questions: call the 'recommend' tool IMMEDIATELY, do NOT add any text before or after the tool call.
    - Never mix a text response with a tool call in the same turn.

    If the user asks outside the context of product recommendations, politely inform them that you can only assist with that.
    """,
    exit_conditions=["text"],
    max_agent_steps=20,
)


