# graph_resources.py
from __future__ import annotations
from functools import lru_cache
from typing  import Tuple, Sequence, Optional

# from  langchain_neo4j import Neo4jGraph
from langchain_community.graphs import Neo4jGraph  # ← inherits GraphStore
from langchain_experimental.graph_transformers import LLMGraphTransformer  # docs 
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

from langchain_core.documents import Document  # for RAG (retrieval-augmented generation)

from llm_config import MAIN_LLM

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
# _llm    = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)

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
        llm=MAIN_LLM,
        allowed_nodes=list(nodes),
        allowed_relationships=list(rels),
        # node_properties=list(props),
        ignore_tool_usage=True, 
    )

@lru_cache(maxsize=32)              # one QA chain per schema triple
def build_rag_chain(key: Tuple[Tuple[str, ...], ...]) -> GraphCypherQAChain:
    transformer = build_transformer(key)      # ensures same key

    cypher_prompt = PromptTemplate(
        template="""You are an expert at generating Cypher queries for Neo4j.
        Use the following schema to generate a Cypher query that answers the given question.
        Make the query flexible by using case-insensitive matching and partial string matching where appropriate.
        Focus on searching paper titles as they contain the most relevant information.
        
        Schema:
        {schema}
        
        Question: {question}
        
        Cypher Query:""",
        input_variables=["schema", "question"],
    )


    qa_prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. 
        Use the following Cypher query results to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise. If topic information is not available, focus on the paper titles.
        
        Question: {question} 
        Cypher Query: {query}
        Query Results: {context} 
        
        Answer:""",
        input_variables=["question", "query", "context"],
    )
    # prompts can be customised here; omitted for brevity
    return GraphCypherQAChain.from_llm(
        graph=_driver,
        cypher_llm=MAIN_LLM,
        qa_llm=MAIN_LLM,
        validate_cypher=True,
        allow_dangerous_requests=True,
        verbose=True,
        return_intermediate_steps=True,
        return_direct=True,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
    )

# 4. public facade
_ingested_keys: set[Tuple[Tuple[str, ...], ...]] = set()  

# ----------------------------------------------------------------------
# 5. Public entry point
# ----------------------------------------------------------------------
def get_resources(nodes: Sequence[str],
                  rels:  Sequence[str],
                  props: Sequence[str], 
                 doc_splits : Optional[Sequence[Document]] = None,  # optional document splits for RAG 
                  ) -> tuple[LLMGraphTransformer, GraphCypherQAChain]:
    
    """Return a transformer and RAG chain for the given schema triple."""

    k = _key(nodes, rels, props)

    

    if doc_splits and k not in _ingested_keys:
        # If document splits are provided, use them for RAG
        graph_transformer = build_transformer(k)

        _driver.add_graph_documents(
            graph_transformer.convert_to_graph_documents(doc_splits)  # transform splits to graph nodes
        )
        _ingested_keys.add(k)

    
    
    
    return build_transformer(k), build_rag_chain(k)
