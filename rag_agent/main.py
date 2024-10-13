from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.chat_models import ChatOllama
from langchain_groq import ChatGroq

from utils import chatbot_with_tools, adv_agentic_rag #MyRag, save_sample_db, db_read_test, sql_agent, 

from dotenv import load_dotenv
load_dotenv()

from functools import partial
def create_prompt(template, **kwargs):
    return str(template).format(**kwargs)

template = '''you are an smart AI assistant.
    Generate compact and summarized answer to {query} with numbering kindly and shortly.
    if there are not enough information to generate answers, just return "Please give me more information" or ask a question for additional information.
    for example, 'could you give me more detailed information about it?'
    '''

def offline_openchat(question:str, template:str = template, llm_name:str="llama3.2"):
    create_greeting_prompt = partial(create_prompt, template)
    prompt = create_greeting_prompt(query=question)
    prompt = ChatPromptTemplate.from_template(prompt)
    query = {"query": question}
    llm = ChatOllama(base_url="http://ollama:11434", model=llm_name)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(query)
    return response

# def online_openchat(question:str, template:str=template):
#     create_greeting_prompt = partial(create_prompt, template)
#     prompt = create_greeting_prompt(query=question)
#     prompt = ChatPromptTemplate.from_template(prompt)
#     query = {"query": question}
#     llm = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")
#     chain = prompt | llm | StrOutputParser()
#     response = chain.invoke(query)
#     return response

def online_agent_chatbot(question:str):
    response = chatbot_with_tools(user_input=question)
    if len(response) >=2: response = f"{response[-1]} \n\n '>>> Source' : {response[-2]}"
    else: response = f"{response[-1]}"
    return response

def open_chat(question:str, template:str=template, llm_name:str="llama3.2"):
    try: return online_agent_chatbot(question=question)
    except:return offline_openchat(template=template, question=question, llm_name=llm_name)


if __name__ == "__main__":

    # save_sample_db()

    # db_read_test()

    # result = sql_agent("2009년에 가장 많이 팔린 장르는 무엇이며, 해당 장르의 총 매출액은 얼마인가요?")
    # print(f"응답개수: {len(result)}")
    # print(result)
    # print("-"*70)
    # print(result[-1])


    res = adv_agentic_rag(user_input="what is the noon report in iss system?")
    print(res)



    pass