import streamlit as st
import requests
import pickle
import random
import asyncio
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosedError

st.set_page_config(page_title="AI Captain", layout="wide")
st.markdown(
            """
        <style>
            .st-emotion-cache-janbn0 {
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

def multi(question:str, recursion_limit:int):
    url = "http://rag_agent:8000/multi_agent"
    json={"question": str(question), "recursion_limit":int(recursion_limit)}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def web(question:str, recursion_limit:int):
    url = "http://rag_agent:8000/web_agent"
    json={"question": str(question), "recursion_limit":int(recursion_limit)}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def sim(question:str, recursion_limit:int):
    url = "http://rag_agent:8000/sim_agent"
    json={"question": str(question), "recursion_limit":int(recursion_limit)}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def rag(question:str, recursion_limit:int):
    url = "http://rag_agent:8000/rag_agent"
    json={"question": str(question), "recursion_limit":int(recursion_limit)}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def sql(question:str, recursion_limit:int):
    url = "http://rag_agent:8000/sql_agent"
    json={"question": str(question), "recursion_limit":int(recursion_limit)}
    response = requests.post(url, json=json)
    res = response.json()
    return res

def filenames(db_path:str="./db/chroma_db_02"):
    url = "http://rag_agent:8000/filenames"
    json={"db_path": db_path}
    response = requests.get(url, json=json)
    res = response.json()
    return res

async def ws_check_random_number():
    uri2 = "ws://rag_agent:8000/ws/random-number"
    async with websockets.connect(uri2) as ws:
        placeholder = st.empty()
        while True:
            try:
                # Receive message from the WebSocket
                message = await ws.recv()
                # Display the message in Streamlit
                placeholder.markdown(f"### {message}")
            except websockets.ConnectionClosed:
                st.write("Connection closed.")
                break



if "messages" not in st.session_state:   st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
if "time_delta" not in st.session_state:   st.session_state.time_delta = ""
if "doc_list" not in st.session_state:   st.session_state.doc_list = ""  
if "check_monitoring" not in st.session_state:   st.session_state.check_monitoring = False  


if __name__ == "__main__":
    st.title("AI CAPTAIN")
    st.session_state.multiple_ws = st.checkbox("Monitoring Random Number")
    if st.session_state.multiple_ws:
        asyncio.run(ws_check_random_number())

    st.markdown("---")


    text_input = st.chat_input("Say something")
    tab1, tab2, tab3, tab4 = st.tabs(["Type1(All in One)", "Type2", "Documents", "Admin"])
    with tab1: 
        with st.spinner("Processing..."):
            if text_input:
                start_time = datetime.now()
                st.session_state.messages.append({"role": "user", "content": text_input})
                result = multi(question= str(text_input), recursion_limit=5)
                st.session_state.messages.append({"role": "assistant", "content": result})
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

        col21, col22, col23, col24 = st.columns(4)
        with col21: sim_check = st.checkbox("Similarity Search", value=True)
        with col22: rag_check = st.checkbox("Rag Search", value=True)
        with col23: web_check = st.checkbox("Web Search", value=True)
        with col24: sql_check = st.checkbox("Sql Search", value=True)

        text_input2 = st.text_input("Say Something...")
        with st.spinner("Processing..."):
            if text_input2:
                if sim_check: 
                    with st.expander("Similarity Search", expanded=True):
                        try:
                            result = sim(question= str(text_input2), recursion_limit=5)
                            with st.container(border=True): 
                                st.markdown(f":green[{result["documents"]}]")
                        except Exception as e:
                            st.info(f"Error: {e}")

                if rag_check: 
                    with st.expander("Rag Search", expanded=True):
                        try:
                            result = rag(question= str(text_input2), recursion_limit=5)
                            with st.container(border=True): 
                                st.markdown(f"{result["generation"]}")
                                st.markdown(f":green[{result["documents"]}]")
                        except Exception as e:
                            st.info(f"Error: {e}")

                if web_check: 
                    with st.expander("Web Search", expanded=True):
                        try:
                            result = web(question= str(text_input2), recursion_limit=5)
                            with st.container(border=True): 
                                st.markdown(f"{result["generation"]}")
                                st.markdown(f":green[{result["documents"]}]")
                        except Exception as e:
                            st.info(f"Error: {e}")

                if sql_check: 
                    with st.expander("SQL Search", expanded=True):
                        try:
                            result = sql(question= str(text_input2), recursion_limit=5)
                            with st.container(border=True): 
                                st.markdown(f"{result["generation"]}")
                                st.markdown(f":green[{result["sql_query"]}]")
                                st.markdown(f"{result["query_rows"]}")
                        except Exception as e:
                            st.info(f"Error: {e}")

    with tab3:
        with st.spinner("Processing..."):
            if st.button("Update Documents List"):
                file_list= filenames(db_path="./db/chroma_db_02")
                with open(file='doc_list.pickle', mode='wb') as f:
                    pickle.dump(file_list, f)

        try:
            with open(file='doc_list.pickle', mode='rb') as f:
                st.session_state.doc_list=pickle.load(f)
            st.session_state.doc_list = [word.upper() for word in st.session_state.doc_list]
            
            input_keywords = st.text_input("Document Keywords(Splitted by Comma)")
            splitted_keywords = input_keywords.split(",")
            splitted_keywords = [word.strip().upper() for word in splitted_keywords]
            with st.expander("Document List"):
                with st.container(border=True):
                    if input_keywords:
                        st.session_state.doc_list = [sentence for sentence in st.session_state.doc_list if all(keyword in sentence for keyword in splitted_keywords)]
                    else: pass

                    for doc in st.session_state.doc_list:
                        with st.container(border=True):
                            st.markdown(f"📘 {doc}")

        except: st.info("Update Documents List")

    with tab4:
        col31, col32 = st.columns(2)
        with col31:
            st.markdown("### Parameter Settings")
            retrieval_k = st.number_input("Count of Retrieved Documnets", min_value=1, max_value=5, value=3)
            agent_recursion_limit = st.number_input("Agent Recursion Limits", min_value=5, max_value=20, value=5)
            gateway_timeout = st.number_input("Connection Timeout(Seconds)", min_value=60, max_value=300, value=90)
        with col32:
            st.markdown("### API Manager")
            groq_api = st.text_input("Groq API")
            tavily_api = st.text_input("Tavily API")
