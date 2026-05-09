from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack.dataclasses import ChatMessage
from haystack.components.joiners import ListJoiner
from haystack import Pipeline
from typing import List
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.converters import OutputAdapter
from haystack.utils import Secret
from getpass import getpass
import os

os.environ["GROQ_API_KEY"] = getpass("Masukkan Groq API Key Anda: ")

os.environ["MONGO_CONNECTION_STRING"] = getpass("Masukkan MongoDB Connection String Anda: ")

system_message = ChatMessage.from_system("You are a helpful assistant that answers questions based on the provided context.")
user_message_template = """
Answer the question based on the user query:
query:{{query}}
answer:
"""
user_message = ChatMessage.from_user(user_message_template)

pipeline = Pipeline()
pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query"], required_variables=["query"]))
pipeline.add_component("generator", OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"))


pipeline.connect("prompt_builder.prompt", "generator.messages")

while True:
    messages = [system_message, user_message]
    query = input("Please input your question or type 'exit' to quit.\n")
    if query.lower() == "exit":
        break
    res = pipeline.run(
        {
            "prompt_builder": {
                "query": query,
                "template":messages
            }
        }
    )
    print("AI Response:", res["generator"]["replies"][0].text)

memory_store = InMemoryChatMessageStore()
memory_retriever = ChatMessageRetriever(memory_store)
memory_writer = ChatMessageWriter(memory_store)

system_message = ChatMessage.from_system("You are a helpful assistant that answers questions based on the provided context.")
user_message_template = """
Answer the question based on the user query, please pay attention to the chat history:
chat_history:
{% for memory in memories %}
    {{memory.text}}
{% endfor %}

query:{{query}}
answer:
"""
user_message = ChatMessage.from_user(user_message_template)

pipeline = Pipeline()
pipeline.add_component("prompt_builder", ChatPromptBuilder(variables=["query","memories"], required_variables=["query","memories"]))
pipeline.add_component("generator", OpenAIChatGenerator(model="llama-3.3-70b-versatile", api_key=Secret.from_token(os.environ["GROQ_API_KEY"]), api_base_url="https://api.groq.com/openai/v1"))
pipeline.add_component("joiner", ListJoiner(List[ChatMessage]))
pipeline.add_component("memory_retriever", memory_retriever)
pipeline.add_component("memory_writer", memory_writer)

pipeline.connect("prompt_builder.prompt", "generator.messages")
pipeline.connect("generator.replies", "joiner")
pipeline.connect("joiner", "memory_writer")
pipeline.connect("memory_retriever", "prompt_builder.memories")

while True:
    messages = [system_message, user_message]
    query = input("Please input your question or type 'exit' to quit.\n")
    if query.lower() == "exit":
        break
    res = pipeline.run(
        data={
            "prompt_builder": {
                "query": query,
                "template":messages
            },
            "joiner":{
                "values": [ChatMessage.from_user(query)]
            },
            "memory_retriever": {
                "chat_history_id": "default"
            },
            "memory_writer": {
                "chat_history_id": "default"
            }
        },
        include_outputs_from=["generator"]
    )
    # print(res)
    print("AI Response:", res["generator"]["replies"][0].text)


