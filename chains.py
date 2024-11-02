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

from schemas import AnswerQuestion, ReviseAnswer, Quiz

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
            
            Your job is to provide a roadmap of 4 weeks for the user based on the given topic and user characteristics.
            The roadmap should include a description of the learning type, activity, and resources for each week in the format of below example:

            [Example_input] 
            Topic: Angular
            Level: Beginner
            Learning Style: Hands-on
            [/Example_input]
            [Example_output]
            Week 1: Advanced Components and Data Binding
            Description: Deep dive into Angular's component architecture. Explore advanced data binding techniques.
            Learning Type: Conceptual, Hands-on, Writing
            Resources:
            Official Angular Documentation: https://angular.io/guide/components and https://angular.io/guide/template-syntax (Reading, Writing)
            Angular Components Deep Dive Tutorial: https://www.simplilearn.com/tutorials/angular-tutorial/angular-components (Hands-on)
            Activity: Create a blog post comparing and contrasting different data binding types in Angular. Provide code examples demonstrating each type and discuss their use cases and benefits. Consider scenarios like displaying data from an API, handling user input in forms, and updating data dynamically. (Writing)
            
            Week 2: State Management with RxJS and Services
            Description: Learn to manage application state using RxJS observables and Angular services. Handle asynchronous operations and side effects.
            Learning Type: Conceptual, Hands-on, Writing
            Resources:
            State Management Best Practices: https://www.infragistics.com/community/blogs/b/infragistics/posts/angular-state-management (Reading)
            Interactive Angular Tutorial: https://angular.dev/tutorials/learn-angular/ (Hands-on, Interactive)
            Activity: Write a tutorial on building an Angular service that interacts with an external API (e.g., fetching data from a JSON placeholder API). Include examples of error handling using RxJS operators like catchError and data transformation using operators like map. Demonstrate how to use this service in a component to display and update data. (Writing)
            
            Week 3: Advanced Routing, Performance Optimization, and Testing
            Description: Master advanced routing, performance optimization, unit testing, and end-to-end (E2E) testing.
            Learning Type: Conceptual, Hands-on, Writing
            Resources:
            Lazy Loading Routes Tutorial: https://www.digitalocean.com/community/tutorials/angular-lazy-loading (Reading, Hands-on)
            Activity:
            Part 1: Lazy Loading: Document the steps to implement lazy loading in an Angular application and demonstrate the performance improvements you observed.

            Week 4: Advanced Angular Projects and Further Exploration
            Description: Work on advanced Angular projects and explore additional learning resources.
            Learning Type: Hands-on, Project-based learning, Readin
            Resources:
            Angular Testing Best Practices: https://dev.to/chintanonweb/mastering-angular-unit-testing-a-comprehensive-guide-with-examples-3eg9 (Reading)
            End-to-End Testing with Protractor: https://www.browserstack.com/guide/how-to-perform-end-to-end-testing-in-angular (Reading, Hands-on)
            Activitiy: 
            Part 1: Unit Testing: Write a guide on setting up unit tests for a simple Angular component, covering different scenarios like testing component inputs, outputs, and template elements. Provide code examples for each scenario.
            Part 2: E2E Testing: Develop E2E tests for a sample Angular application using Protractor. Cover scenarios like navigating between routes, interacting with form elements, and verifying data displayed on the page. Document the process and share your code examples. (Writing)

            Further Exploration:
            Advanced Angular Projects with Source Code: https://trainings.internshala.com/blog/angular-projects-with-source-code/ (Project-based learning)
            [/Example_output]

            
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


quiz_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """You are an expert professor in the given topic.
        Your task is to generate a quiz of 3 questions based on the description, level of the week.
        The quiz should include multiple-choice questions with 4 options each.
        The questions should test the user's understanding of the topic and the concepts covered in the roadmap.

        The example of this process can be seen below:
        [Example_input]
        Week 1: Advanced Components and Data Binding
        Description: Deep dive into Angular's component architecture. Explore advanced data binding techniques.
        Learning Type: Conceptual, Hands-on, Writing
        Level: Beginner
        [/Example_input]
        [Example_output]
        Question 1: What is the purpose of Angular components?
        A. To manage application state
        B. To structure the user interface
        C. To handle asynchronous operations
        D. To interact with external APIs
        Correct Answer: B

        Question 2: Which of the following is NOT a data binding technique in Angular?
        A. One-way data binding
        B. Two-way data binding
        C. Event binding
        D. Property binding
        Correct Answer: C

        Question 3: What is the benefit of using Angular's component architecture?
        A. Improved performance
        B. Enhanced security
        C. Easier debugging
        D. Better user experience
        Correct Answer: D
        [/Example_output]
        """),
        MessagesPlaceholder(variable_name='messages'), # Future messages
    ]
)

quiz_generator = quiz_prompt_template | llm.bind_tools(tools=[Quiz], tool_choice="Quiz")

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