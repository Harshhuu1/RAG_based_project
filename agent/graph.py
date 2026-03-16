from typing import TypedDict , Annotated , List
from loguru import logger
import operator

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from agent.tools import TOOLS
from agent.prompts import SYSTEM_PROMPT
from config import settings , agent_cfg

# _____Agent state______
class AgentState(TypedDict):
    """state that flows through the agent graph"""
    messages: Annotated[List[BaseMessage],operator.add]
    query:str
    final_answer:str

# __agent nodes______

def agent_node(state:AgentState)->AgentState:
    """Main agent node- decides which tool to use."""
    logger.info("Agent node called...")

    llm=ChatOpenAI(
        model=agent_cfg.model,
        temperature=agent_cfg.temperature,
        max_tokens=agent_cfg.max_tokens,
        openai_api_key=settings.openai_api_key,

    ).bind_tools(TOOLS)

    messages=[SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response=llm.invoke(messages)

    logger.info(f"Agent response: {response.type}")
    return {"messages":[response]}
def should_continue(state:AgentState)->str:
    """decide whether to call a tool or end"""
    last_message = state["messages"][-1]
    
    if hasattr(last_message,"tool_calls") and last_message.tool_calls:
        logger.info(f"Tool called:{last_message.tool_calls[0]['name']}")
        return "tools"
    logger.info("Agent finished -no more tool calls")
    return END

#__build graph____

def build_graph():
    """Build and return the Langraph agent"""

    #initialize graph with state

    graph=StateGraph(AgentState)

    #add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools",ToolNode(TOOLS))

    #set entry point

    graph.set_entry_point("agent")

    #add edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools":"tools",
            END:END,

        }
    )
    graph.add_edge("tools","agent")
    return graph.compile()


#___Main query function

def query_agent(question: str, chat_history: list = []) -> str:
    """send a question to the agent and get an answer"""
    logger.info(f"Query received: '{question[:50]}'")

    graph = build_graph()

    # Build messages with chat history
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))

    initial_state = {
        "messages": messages,
        "query": question,
        "final_answer": "",
    }

    result = graph.invoke(initial_state)
    final_message = result["messages"][-1]

    logger.success("Agent query complete")
    return final_message.content
# What this means:

# add_node → registers each node in the graph
# set_entry_point("agent") → agent node always runs first
# add_conditional_edges → after agent node, go to tools or END based on should_continue
# add_edge("tools", "agent") → after tools run, always go back to agent. This creates the loop that lets agent call multiple tools!
# query_agent() → simple function to call the whole graph with one line

# Type it and tell me when done! 

