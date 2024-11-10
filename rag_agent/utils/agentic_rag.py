import re
import requests
import chromadb
import numpy as np
from typing import Literal
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


try:
    llm = ChatGroq(temperature=0, model_name= "llama-3.2-11b-text-preview")
    print(f">>> {llm}")
except:
    llm = ChatOllama(base_url="http://ollama:11434", model="llama3.2:latest", temperature=0)
    print(f">>> {llm}")

db_path = "./db/chroma_db_02"
vectorstore = Chroma(collection_name="collection_01", persist_directory=db_path, embedding_function=OllamaEmbeddings(base_url="http://ollama:11434", model="bge-m3:latest"))

#### Helper Function ##############################
def vectordb_targets(db_path:str):
    client = chromadb.PersistentClient(path=db_path)
    for collection in client.list_collections():
        data = collection.get(include=['metadatas'])
    lv1 = list(set([d['First Division'] for d in data["metadatas"]]))
    lv2 = list(set([d['Second Division'] for d in data["metadatas"]]))
    rag_target = lv1 + lv2
    rag_target.insert(0, "vectorstore")
    rag_target.insert(0, "vectordb")
    docs = ", ".join(rag_target)
    return docs

docs = vectordb_targets(db_path=db_path)

def get_semantic_search_docs(query:str, vectorstore, k:int=100, fetch_k:int=200):
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={'k': k, "fetch_k": fetch_k}
        )
    result = retriever.invoke(query)
    return result

def get_bm25_top_docs(query:str, documents:list, top_k:int=20):
    tokenized_corpus = [doc.page_content for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    pattern = r'`(.*?)`'  # 백틱으로 둘러싸인 단어만 검색 대상으로 리스트에 담기
    extracted_keywords = re.findall(pattern, query)
    doc_scores = bm25.get_scores(extracted_keywords)
    sorted_indices = np.argsort(doc_scores)  # 값에 대한 정렬된 인덱스
    order_values = np.empty_like(sorted_indices)
    order_values[sorted_indices] = np.arange(len(doc_scores))
    top_index = [i for i, value in enumerate(order_values) if value < top_k]
    top_docs = [i for i in documents if documents.index(i) in top_index ]
    return top_docs
    
def get_keywords_matched_docs(query:str, documents:list, and_condition:bool=True):
    pattern = r'`(.*?)`'  # 백틱으로 둘러싸인 단어만 검색 대상으로 리스트에 담기
    extracted_keywords = re.findall(pattern, query)
    lower_keywors = [keyword.lower() for keyword in extracted_keywords]
    lower_docs = [doc.page_content.lower() for doc in documents]
    if and_condition: matching_sentences = [sentence for sentence in lower_docs if all(keyword in sentence for keyword in lower_keywors)]
    else: matching_sentences = [sentence for sentence in lower_docs if any(keyword in sentence for keyword in lower_keywors)]
    matched_index = [lower_docs.index(doc) for doc in matching_sentences]
    final_matched_docs = [documents[i] for i in matched_index]
    return final_matched_docs

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["similarity_search", "vectorstore", "web_search", "database"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore or a similarity or a database.",
    )

# Prompt
system = f"""You are an expert at routing a user question to a vectorstore, web search or database.
The vectorstore contains documents related to {docs}, Use the vectorstore for questions on these topics. 
The question contains words of similarity or sim search, Use similarity_search for the question.
The question contains words related to database, Use the database for the question. 
Otherwise, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
structured_llm_router = llm.with_structured_output(RouteQuery)
question_router = route_prompt | structured_llm_router


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)
structured_llm_grader = llm.with_structured_output(GradeDocuments)
retrieval_grader = grade_prompt | structured_llm_grader

prompt = ChatPromptTemplate.from_messages([
    ("human", 
    """You are a smart AI assistant. 
    Use the following pieces of retrieved context to answer the question. 
    Generate detailed answer including specified numbers, fomulas in the point of technical specifications. 
    If you don't know the answer, just say that you don't know. 
    Question: {question} 
    Context: {context} 
    Answer:"""),
    ])

rag_chain = prompt | llm | StrOutputParser()

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)
hallucination_grader = hallucination_prompt | structured_llm_grader

class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)
structured_llm_grader = llm.with_structured_output(GradeAnswer)
answer_grader = answer_prompt | structured_llm_grader


system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)
question_rewriter = re_write_prompt | llm | StrOutputParser()


from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=3)


### Construct Graph #######################################
from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        connection: web connection
        question: question
        generation: LLM generation
        documents: list of documents
    """
    connection: bool = False
    question: str
    generation: str
    documents: List[str]

from langchain.schema import Document

# Nodes

def similarity_search(state):
    """
    do hybrid similarity search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation

    """
    print("---Similarity Search---")
    question = state["question"]

    # Retrieval
    pattern = r'"(.*?)"'  # 따옴표로 둘러싸인 단어만 검색 대상으로 리스트에 담기
    extracted_keywords = re.findall(pattern, question)
    if len(extracted_keywords) > 0:
        documents = get_semantic_search_docs(query=question, vectorstore=vectorstore, k=100, fetch_k=500)
        print(f">>> 1st retrieved docs counts : {len(documents)}")
        documents = get_bm25_top_docs(query=question, documents=documents, top_k=10)
        print(f">>> 2rd retrieved docs counts : {len(documents)}")
        documents = get_keywords_matched_docs(query=question, documents=documents, and_condition=True)
        print(f">>> 3rd retrieved docs counts : {len(documents)}")
    else: documents = get_semantic_search_docs(query=question, vectorstore=vectorstore, k=3, fetch_k=500)
    
    return {"documents": documents, "question": question}

def sql_search(state):
    """
    Generate answer based on Databse

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation

    """
    print("---SQL SEARCH---")
    question = state["question"]
    return {"documents": "Under Construction", "question": question, "generation": "Under Construction"}


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    pattern = r'"(.*?)"'  # 따옴표로 둘러싸인 단어만 검색 대상으로 리스트에 담기
    extracted_keywords = re.findall(pattern, question)
    if len(extracted_keywords) > 0:
        documents = get_semantic_search_docs(query=question, vectorstore=vectorstore, k=100, fetch_k=500)
        documents = get_bm25_top_docs(query=question, documents=documents, top_k=10)
        documents = get_keywords_matched_docs(query=question, documents=documents, and_condition=True)
    else: documents = get_semantic_search_docs(query=question, vectorstore=vectorstore, k=3, fetch_k=500)
    
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    question = state["question"]
    documents = state["documents"]

    # Score each doc
    try:
        print(f"Count of Retrieved Docs: {len(documents)}")
        filtered_docs = []
        for d in documents:
            score = retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}
    except Exception as e:
        print(f"Count of Retrieved Docs: {len(documents)}")
        print(e)


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")

    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_connection(state):
    print("---WEB CONNECTION---")
    res = check_internet(state)
    return {"connection": res}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges ###

def check_internet(state):
    """Check internet connection by pinging a website."""
    
    print("---CHECK INTERNET CONNECTION---")
    # question = state["question"]
    try:
        response = requests.get("https://www.google.com", timeout=2)
        # If the request is successful, we assume internet is ON
        if response.status_code == 200:
            # print("Internet is ON")
            return "ON"
    except requests.ConnectionError:
        # If there is a connection error, we assume internet is OFF
        # print("Internet is OFF")
        return "OFF"
    
    
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    try:
        question = state["question"]
        source = question_router.invoke({"question": question})
        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif source.datasource == "vectorstore":
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"
        elif source.datasource == "similarity_search":
            print("---ROUTE QUESTION TO SIMILARITY SEARCH---")
            return "similarity_search"
        elif source.datasource == "database":
            print("---ROUTE QUESTION TO DATABASE---")
            return "database"
    except Exception as e:
        print(f"---ROUTING ERROR {e}---")
        return e


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION(Useful): GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION(Not Useful): GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION(Not Supported): GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
### Complile Graph #############################

from langgraph.graph import StateGraph, START, END

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_connection", web_connection)  # check_internet
workflow.add_node("similarity_search", similarity_search)  # similarity_search
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("sql_search", sql_search)  # sql search
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_connection",
        "vectorstore": "retrieve",
        "similarity_search": "similarity_search",
        "database": "sql_search"
    },
)

workflow.add_conditional_edges(
    "web_connection",
    check_internet,
    {
        "ON": "web_search",
        "OFF": END,
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("similarity_search", END)
workflow.add_edge("sql_search", END)

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)


# Compile
app = workflow.compile()


### APP STREAM ##########################
from langgraph.errors import GraphRecursionError
# Run
def app_stream(question:str, recursion_limit:int=5):
    inputs = {
        "question": question, 
        }
    config = {
        "recursion_limit": recursion_limit, 
        }
    try:
        for output in app.stream(inputs, 
                                config, 
                                # stream_mode="debug"
                                ):
            for key, value in output.items():
                # Node
                print(f">>> Node : {key}")
            print("="*70)

        # Final generation
        print("")
    except GraphRecursionError:
        print(f"=== Maximum Recursion Error : {recursion_limit} ===")
        value = f"=== Maximum Recursion Error : {recursion_limit} ==="
    
    return value







