from dotenv import load_dotenv
from typing import List
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from schemas import AnswerQuestion, Reflection
from chains import parser
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from collections import defaultdict
import json
import os

load_dotenv()

search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=3)
tool_executor = ToolExecutor([tavily_tool]) # so that we can run different executors in parallel

from dotenv import load_dotenv  
from typing import List  
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage  
from schemas import AnswerQuestion, Reflection  
from chains import parser, parser_pydantic 
from langgraph.prebuilt import ToolInvocation, ToolExecutor  
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper  
from langchain_community.tools.tavily_search import TavilySearchResults  
from collections import defaultdict  
import json  
import os  
  
load_dotenv()  
  
search = TavilySearchAPIWrapper(tavily_api_key=os.getenv("TAVILY_API_KEY"))  
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=3)  
tool_executor = ToolExecutor([tavily_tool])  # so that we can run different executors in parallel  
  
# Tool message represents the result of a tool call. This is distinct from a FunctionMessage in order to match OpenAI's function and tool message types.
def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)

    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                ToolInvocation(
                    tool="tavily_search_results_json",
                    tool_input=query
                ))
            ids.append(parsed_call["id"])

    outputs = tool_executor.batch(tool_invocations) # Run them in batch

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, output_map in outputs_map.items():
        tool_messages.append(
            ToolMessage(
                content=json.dumps(output_map), tool_call_id=id_
            )
        )

    
    return tool_messages

if __name__ == "__main__":
    print("Tool Executor Enter")

    human_message = HumanMessage(
        content="write about AI powered SOC / Autonomous soc problem domain"
        " list startups that do that and raised capital"
        )
    
    # Dummy object of an answer
    answer = AnswerQuestion(
        answer="",
        reflection = Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups"
        ],
        id="call_Kasofaofjafap"
    )

    # Calling executor tools function with dummy state
    raw_res = execute_tools(
        state = [
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_Kasofaofjafap"
                    }
                ]
            )
        ]
    )