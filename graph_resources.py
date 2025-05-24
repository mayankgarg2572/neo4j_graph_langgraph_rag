# graph_resources.py
from __future__ import annotations
from functools import lru_cache
from typing import Tuple, Sequence

from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer  # docs 
from langchain.chains import GraphCypherQAChain                            # docs 
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------------------------------------------------
# 1. Global, reusable driver and LLM (schema-independent)
# ----------------------------------------------------------------------
# I want to access the database uri stored in the environment variable
from dotenv import load_dotenv
import os
load_dotenv()  # load environment variables from .env file
DATABASE_URI = "bolt://localhost:7687"  # default to local Neo4j instance


_driver = Neo4jGraph(
    url=os.getenv("NEO4J_URI", DATABASE_URI),  # default URI
    username=os.getenv("NEO4J_USERNAME", "neo4j"),  # default username
    password=os.getenv("NEO4J_PASSWORD", "12345678"),  # default password
)               # one connection pool per process (thread–safe) :contentReference[oaicite:5]{index=5}
_llm    = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

# ----------------------------------------------------------------------
# 2. Cache-key helper
# ----------------------------------------------------------------------
def _key(nodes: Sequence[str],
         rels:  Sequence[str],
         props: Sequence[str]) -> Tuple[Tuple[str, ...], ...]:
    """Return a hashable, order-insensitive key for a schema triple."""
    return tuple(map(tuple, (sorted(nodes), sorted(rels), sorted(props))))

# ----------------------------------------------------------------------
# 3. Cached builders
# ----------------------------------------------------------------------
@lru_cache(maxsize=32)              # threadsafe memoisation :contentReference[oaicite:6]{index=6}
def build_transformer(key: Tuple[Tuple[str, ...], ...]) -> LLMGraphTransformer:
    nodes, rels, props = key
    return LLMGraphTransformer(
        llm=_llm,
        allowed_nodes=list(nodes),
        allowed_relationships=list(rels),
        node_properties=list(props),
    )

@lru_cache(maxsize=32)              # one QA chain per schema triple
def build_rag_chain(key: Tuple[Tuple[str, ...], ...]) -> GraphCypherQAChain:
    transformer = build_transformer(key)      # ensures same key
    # prompts can be customised here; omitted for brevity
    return GraphCypherQAChain.from_llm(
        graph=_driver,
        cypher_llm=_llm,
        qa_llm=_llm,
        validate_cypher=True,
        allow_dangerous_requests=True,
    )

# ----------------------------------------------------------------------
# 4. Public entry point
# ----------------------------------------------------------------------
def get_resources(nodes: Sequence[str],
                  rels:  Sequence[str],
                  props: Sequence[str]) -> tuple[LLMGraphTransformer, GraphCypherQAChain]:
    k = _key(nodes, rels, props)
    return build_transformer(k), build_rag_chain(k)
