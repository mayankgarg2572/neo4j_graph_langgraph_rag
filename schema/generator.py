import json
from pydantic import BaseModel
# from llm_config import SCHEMA_LLM
from config import  SCHEMA_LLM
class GraphSchema(BaseModel):
    allowed_nodes: list[str]
    node_properties: list[str]
    allowed_relationships: list[str]


def generate_schema(docs) -> GraphSchema:
    print("In function generate_schema with args, docs:", docs)
    prompt = (
       "You are a graph schema expert. Analyze the following documents and extract:\n"
        "- The most appropriate node types (entity categories)\n"
        "- Node properties (metadata fields)\n" 
        "- Possible relationships between entities\n\n"
        "Return ONLY valid JSON in this structure:\n"
        "{\n"
        '  "allowed_nodes": ["Entity1", "Entity2"],\n'
        '  "node_properties": ["title", "summary"],\n'
        '  "allowed_relationships": ["AUTHORED", "MENTIONS"]\n'
        "}\n\n"
        "No need to provide any explanations or additional text.\n\n"
        "Ensure all keys are present and all values are lists of strings as in above example.\n\n"
        "Must ensure you just return a valid JSON object having structure similar to the above example with no additional text.\n\n"
        "Documents:\n"
        + "\n\n".join(d.page_content for d in docs)
    )
    resp = SCHEMA_LLM.invoke(prompt)
    try:
        data = json.loads(resp.content if hasattr(resp, "content") else resp)
        return GraphSchema(**data)
    except Exception:
        # Return empty schema; caller decides what to do
        return GraphSchema(allowed_nodes=[], node_properties=[], allowed_relationships=[])
