from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, RetrievalQA
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field
import json
from typing import Literal, List
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from langchain import hub
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from pprint import pprint



class MyRag:
    def rag_chat(query:str, json_style:bool=True):

        embed_model = OllamaEmbeddings(base_url="http://ollama:11434", model="bge-m3:latest")
        db_path = "./db/chroma_langchain_db"
        vectorstore = Chroma(collection_name="my_collection", persist_directory=db_path, embedding_function=embed_model)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})

        if json_style:
            system_prompt = ('''
        You are a knowledgable shipbuilding engineer for technical question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        Generate detailed answer including specified numbers, fomulas in the point of technical specifications. 
        If you don't know the answer, just say that you don't know. 

        {context}
        Please provide your answer in the following JSON format: 
        {{
        "answer": "Your detailed answer here",\n
        "keywords: [list of important keywords from the context] \n
        "sources": "Direct sentences or paragraphs from the context that support your answers. ONLY RELEVANT TEXT DIRECTLY FROM THE DOCUMENTS. DO NOT ADD ANYTHING EXTRA. DO NOT INVENT ANYTHING."
        }}
        The JSON must be a valid json format and can be read with json.loads() in Python. Answer:
                            ''')
        
        else: 
            system_prompt = ("""
        You are a knowledgable shipbuilding engineer for technical question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        Generate detailed answer including specified numbers, fomulas in the point of technical specifications. 
        If you don't know the answer, just say that you don't know. 

        {context}
        """
                )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
                ]
            )

        try:
            model = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": f"{query}"})
            return response["context"], response["answer"]
        except:
            model = ChatOllama(temperature=0, base_url="http://ollama:11434", model= "llama3.2:latest")
            question_answer_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": f"{query}"})
            return response["context"], response["answer"]
 
