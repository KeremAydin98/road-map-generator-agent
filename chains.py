import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser
)
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from schemas import AnswerQuestion, ReviseAnswer

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

parser = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

# Actor Agent Prompt Template
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert researcher.
            Current time: {time}
            
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer.
            
            Answer the user's question above using the required format. Your response MUST be less than 500 words."""
        ),
        MessagesPlaceholder(variable_name='messages'), # Future messages
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

# Roadmap Generator Agent Prompt Template
roadmap_generator_prompt_template = actor_prompt_template.partial(
    first_instruction="Based on the user characteristics and the given a topic provide a list of subjects to learn to understand the given topic."
)
roadmap_generator = roadmap_generator_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

# Reviser Agent Prompt Template
revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST provide links in your revised answer to ensure it can be verified.
        - For each source the roadmap MUST provide description, link and learning type like this.
            - Description: A brief description of the source.
            - Link: The URL of the source.
            - Learning Type: The type of learning the source provides.
    - You should use the previous critique to remove superfluous information from your roadmap."""

revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == '__main__':

    # Define user characteristics
    input_data = {
        'topic': 'Kubernetes',
        'level': 'Junior',
        'learning_style': 'Reading',
        'time_frame': '3 weeks',
        'schedule_type': 'Weekly',
    }

    # Message to pass into chain
    human_message = HumanMessage(
        content="Based on the user characteristics and the given topic, provide me a roadmap."
    )

    # Chain with populated inputs
    chain = (
        roadmap_generator_prompt_template | 
        llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    )

    # Invoke the chain with `input_data` as parameters
    res = chain.invoke({**input_data, 
                        'messages': [human_message],
                        'input_language': 'English',
                        'output_language': 'English'})
    print(res)