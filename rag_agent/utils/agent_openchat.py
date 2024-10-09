
#### [Start] Basic Chatbot
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def basic_chatbot():

    graph_builder = StateGraph(State)
    print(graph_builder)
    print("-"*70)

    model = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")

    def chatbot(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    graph = graph_builder.compile()


    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [("user", user_input)]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


#### [Start] Chatbot with tools ################################
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated, Literal
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
from langchain_core.messages import ToolMessage
from langchain_core.messages import RemoveMessage

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# def filter_messages(state:State):    # memory에 최근 2개의 질문 이력만 남기기
#     delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
#     return {"messages": delete_messages}


def chatbot_with_tools(user_input:str):
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    tool = TavilySearchResults(max_results=2)
    tools = [tool]
    # tool.invoke("What's a 'node' in LangGraph?")


    llm = ChatGroq(temperature=0, model_name= "llama-3.2-90b-text-preview")
    # Modification: tell the LLM which tools it can call
    llm_with_tools = llm.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    graph_builder = StateGraph(State)

    # graph_builder.add_node("filter", filter_messages)
    graph_builder.add_node("chatbot", chatbot)


    tool_node = BasicToolNode(tools=[tool])
    graph_builder.add_node("tools", tool_node)



    def route_tools(state: State,):
        """
        Use in the conditional_edge to route to the ToolNode if the last message
        has tool calls. Otherwise, route to the end.
        """
        if isinstance(state, list):
            ai_message = state[-1]
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state to tool_edge: {state}")
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
            return "tools"
        return END



    # The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
    # it is fine directly responding. This conditional routing defines the main agent loop.

    # Any time a tool is called, we return to the chatbot to decide the next step
    # graph_builder.add_edge(START, "filter")
    # graph_builder.add_edge("filter", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
        )
    graph_builder.add_edge("tools", "chatbot")
    graph = graph_builder.compile(checkpointer=memory)

    
    def stream_graph_updates(user_input: str):
        all_result = []
        config = {"configurable": {"thread_id": "1"}}
        for event in graph.stream({"messages": [("user", user_input)]}, config=config, debug=False):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)
                all_result.append(value["messages"][-1].content)
            
        return all_result

    
    result = stream_graph_updates(user_input)
    return result
       
        





if __name__ =="__main__":

    # basic_chatbot()

    res = chatbot_with_tools("check the weather condition of Korea yesterday")
    print(">>>>>>>>>>>>>>>>>>>>")
    print(res[-2])
    print(res[-1])
    pass