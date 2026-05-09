from haystack import Pipeline, component
from pymongo import MongoClient
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
import os
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage
from typing import List
from dotenv import load_dotenv
load_dotenv()

# os.environ['MONGO_CONNECTION_STRING'] = getpass("Enter your MongoDB connection string: ")

# os.environ['GROQ_API_KEY'] = getpass("Enter your Groq API key: ")

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

TEMPLATE = """
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
5. Input: {{input}}
output:

```

"""

pipeline = Pipeline()
pipeline.add_component("materials", GetMaterials())
pipeline.add_component("categories", GetCategories())
pipeline.add_component(
    "prompt_builder",
    ChatPromptBuilder(
        variables=["input", "materials", "categories"],
        required_variables=["input", "materials", "categories"],
    )
)
pipeline.add_component("generator", OpenAIChatGenerator(
    model="llama-3.3-70b-versatile",
    api_key=Secret.from_token(os.environ['GROQ_API_KEY']),
    api_base_url="https://api.groq.com/openai/v1"
))

pipeline.connect("materials.materials", "prompt_builder.materials")
pipeline.connect("categories.categories", "prompt_builder.categories")
pipeline.connect("prompt_builder.prompt","generator.messages")

user_input = "I want to find an Outerwear that is not make me hot"

response = pipeline.run(
    {
        "prompt_builder":{
            "input": user_input,
            "template": [ChatMessage.from_user(TEMPLATE)]
        }
    }
)

response

import re
import json
response_selected = response['generator']['replies'][0].text
json_match = re.search(r'```json\n(.*?)\n```', response_selected, re.DOTALL)
if json_match:
    json_string = json_match.group(1)
    # Parse the JSON string into a Python object
    data = json.loads(json_string)
    # print(data)
else:
    print("No JSON found.")

data

response_selected


