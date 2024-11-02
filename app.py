from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langgraph.graph import END, MessageGraph
from typing import List
from chains import revisor, roadmap_generator, quiz_generator
from tool_executor import execute_tools
import re

# Load environment variables
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

# Set up the LangChain graph as in the original code
MAX_ITERATIONS = 3

builder = MessageGraph()
builder.add_node("draft", roadmap_generator)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")

def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    if count_tool_visits > MAX_ITERATIONS:
        return END
    return "execute_tools"

builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")

# Compile the graph
graph = builder.compile()

@app.route('/create_quiz', methods=['POST'])
def create_quiz():
    # Get the request data (JSON input with user characteristics and topic)
    data = request.get_json()
    description = data.get('description', '')
    learningType = data.get('learningType', '')
    level = data.get('level', '')


    if not description:
        return jsonify({"error": "description is required"}), 400
    elif not learningType:
        return jsonify({"error": "learningType is required"}), 400
    elif not level:
        return jsonify({"error": "level is required"}), 400

    content = f"**Description** \
            {description} \
            **Learning Type** \
            {learningType} \
            **Level** \
            {level} \
            Based on the user characteristics and the given topic, provide me a quiz."
    
    human_message = HumanMessage(
        content=content
    )

    try:
        # Quiz generator chain
        response = quiz_generator.invoke({'messages': [human_message], 'input_language': 'English', 'output_language': 'English'})
        questions = []

        for message in response.tools_calls:
            if message.tool_name == "Quiz":
                questions.append({
                    "question": message['args']['question'],
                    "answers": message['args']['options'],
                    "correctAnswer": message['args']['correct_answer']
                })

        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/invoke_graph', methods=['POST'])
def invoke_graph():
    # Get the request data (JSON input with user characteristics and topic)
    data = request.get_json()
    topic = data.get('topic', '')
    level = data.get('level', '')
    learning_style = data.get('learning_style', '')

    # Message to pass into chain
    content=f"**Topic** \
            {topic} \
            **User Characteristics** \
            Level: {level}, \
            Learning style: {learning_style}, \
            Based on the user characteristics and the given topic, provide me a roadmap."

    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    elif not level:
        return jsonify({"error": "Level is required"}), 400
    elif not learning_style:
        return jsonify({"error": "Learning style is required"}), 400
    
    try:
        # Invoke the graph with the user prompt
        response = graph.invoke(content)
        response_text = response[-1].tool_calls[0]['args']['answer']

        # Split data by each week
        weeks = response_text.split('Week')
        weeks = weeks[1:]
        weekly_schedule = {"weeks": {}}

        for week_data in weeks:

            # Extract week number
            week_num = week_data.split(':')[0].strip()

            # Extract description
            description_match = re.search(r"Description: (.*?)\\n", week_data)
            description = description_match.group(1) if description_match else ""

            # Extract learning type
            learning_type_match = re.search(r"Learning Type: (.*?)\\n", week_data)
            learning_types = learning_type_match.group(1).split(", ") if learning_type_match else []

            # Extract activity
            activity_match = re.search(r"Activity: (.*?)\\n", week_data)
            activity = activity_match.group(1) if activity_match else ""

            # Extract resources
            resources = []
            for resource in re.findall(r"- Description: (.*?)\\n- Link: (.*?)\\n- Learning Type: (.*?)\\n", week_data):
                resource_desc, resource_link, resource_type = resource
                resources.append({
                    "description": resource_desc.replace('\\', ''),
                    "link": resource_link.replace('\\', ''),
                    "learningType": resource_type.replace('\\', '').split(", ")
                })

            # Store extracted data in the JSON structure
            weekly_schedule["weeks"][f'Week {week_num}'] = {
                "description": description.replace('\\', ''),
                "learningType": [learning_type.replace('\\', '') for learning_type in learning_types],
                "activity": activity.replace('\\', ''),
                "resources": resources
            }

        return jsonify({"weeklySchedule": weekly_schedule})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
