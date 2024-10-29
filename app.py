from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from typing import List

# Import the graph and required components
from chains import revisor, roadmap_generator
from tool_executor import execute_tools

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

@app.route('/invoke_graph', methods=['POST'])
def invoke_graph():
    # Get the request data (JSON input with user characteristics and topic)
    data = request.get_json()
    topic = data.get('topic', '')
    level = data.get('level', '')
    learning_style = data.get('learning_style', '')
    time_frame = data.get('time_frame', '')
    schedule_type = data.get('schedule_type', '')

    # Message to pass into chain
    content=f"**Topic** \
            {topic} \
            **User Characteristics** \
            Level: {level}, \
            Learning style: {learning_style}, \
            Time frame: {time_frame}, \
            Schedule type: {schedule_type} \
            Based on the user characteristics and the given topic, provide me a roadmap."

    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    elif not level:
        return jsonify({"error": "Level is required"}), 400
    elif not learning_style:
        return jsonify({"error": "Learning style is required"}), 400
    elif not time_frame:
        return jsonify({"error": "Time frame is required"}), 400
    elif not schedule_type:
        return jsonify({"error": "Schedule type is required"}), 400

    try:
        # Invoke the graph with the user prompt
        response = graph.invoke(content)

        return jsonify({"response": response[-1].tool_calls[0]['args']['answer']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
