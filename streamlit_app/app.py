import streamlit as st
import requests
import random
from datetime import datetime

st.set_page_config(page_title="AI Captain", layout="wide")
st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

def calculate_time_delta(start, end):
    delta = end - start
    return delta.total_seconds()

def openchat(query:str):
    url = "http://rag_agent:8000/ask"
    json={"prompt": query}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def similarity_search(query:str):
    url = "http://rag_agent:8000/search"
    json={"prompt": query}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def basic_rag(query:str, json_style:bool="True"):
    url = "http://rag_agent:8000/basic_rag"
    json={"prompt": query, "json_style":json_style}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def agentic_rag(query:str):
    url = "http://rag_agent:8000/agent_rag"
    json={"prompt": query}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def filenames():
    url = "http://rag_agent:8000/filenames"
    json={"db_path": "./db/chroma_db_02"}
    response = requests.get(url, json=json)
    res = response.json()
    return res



if "messages" not in st.session_state:   st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "time_delta" not in st.session_state:   st.session_state.time_delta = ""
if "service" not in st.session_state:   st.session_state.service = ""
if "doc_list" not in st.session_state:   st.session_state.doc_list = []

if __name__ == "__main__":
    st.title("AI CAPTAIN")
    st.markdown("---")
    st.session_state.service = st.radio(label="Selection", options=["Open Chat", "Similarity Search", "Basic Rag", "Agentic Rag"])
    st.write('<style>div.stRadio > div{flex-direction:row; align-itmes: stretch;}</style>', unsafe_allow_html=True)    

    text_input = st.chat_input("Say something")

    tab1, tab2, tab3 = st.tabs(["MAIN", "Documents", "Admin"])
    with tab1:
        with st.spinner("Processing..."):
            if text_input and st.session_state.service == "Open Chat":
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input})
                result = openchat(query = text_input)
                st.session_state.messages.append({"role": "assistant", "content": result})
                end_time = datetime.now()
                st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            if text_input and st.session_state.service == "Similarity Search":
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input})
                result = similarity_search(query = text_input)
                st.session_state.messages.append({"role": "assistant", "content": result})
                end_time = datetime.now()
                st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            elif text_input and st.session_state.service == "Basic Rag":
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input})
                result = basic_rag(query = text_input)
                st.session_state.messages.append({"role": "assistant", "content": result})
                end_time = datetime.now()
                st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            elif text_input and st.session_state.service == "Agentic Rag":
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input})
                try: 
                    result = agentic_rag(query = text_input)
                    st.session_state.messages.append({"role": "assistant", "content": result})
                except Exception as e:
                    st.warning(f"Time Limit exceeds - {e}")
                
                end_time = datetime.now()
                st.session_state.time_delta = calculate_time_delta(start_time, end_time)
            else: pass

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message(msg["role"], avatar="👨‍✈️").write(msg["content"])
            else:
                st.chat_message(msg["role"], avatar="🤖").write(msg["content"])
        if st.session_state.time_delta: 
            st.warning(f"⏱️ TimeDelta(Sec) : {st.session_state.time_delta}")

    with tab2:
        st.session_state.doc_list = filenames()
        with st.container(border=True, height=500):
            for d in st.session_state.doc_list:
                with st.container(border=True):
                    st.markdown(f"📘 {d}")

    with tab3:
        col31, col32 = st.columns(2)
        with col31:
            st.markdown("### Parameter Settings")
            retrieval_k = st.number_input("Count of Retrieved Documnets", min_value=1, max_value=5, value=3)
            agent_recursion_limit = st.number_input("Agent Recursion Limits", min_value=6, max_value=20, value=10)
        with col32:
            st.markdown("### API Manager")
            groq_api = st.text_input("Groq API")
            openai_api = st.text_input("OPEN AI API")
            tavily_api = st.text_input("Tavily API")
