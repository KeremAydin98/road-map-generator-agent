from dotenv import load_dotenv
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.graph import END, StateGraph, MessageGraph

from chains import revisor, roadmap_generator, quiz_generator
from tool_executor import execute_tools

from termcolor import colored

load_dotenv()

MAX_ITERATIONS = 3

class State(TypedDict):
    messages: Annotated[List[str], add_messages]


builder = MessageGraph()
builder.add_node("draft", roadmap_generator)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

def event_loop(state: List[BaseMessage]):
    print(colored(f"Current state in loop: {state}", 'red'))
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return 'execute_tools'

builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == '__main__':

    # Define user characteristics
    input_data = {
        'topic': 'Kubernetes',
        'level': 'Junior',
        'learning_style': 'Reading',
    }

    # Message to pass into chain
    content=f"""**Topic**
            {input_data['topic']}

            **User Characteristics**
            Level: {input_data['level']},
            Learning style: {input_data['learning_style']},
            
            Based on the user characteristics and the given topic, provide me a roadmap."""
    
    human_message = HumanMessage(
        content=content
    )

    # Invoke the chain with `input_data` as parameters
    res = graph.invoke(content)
    print(res)