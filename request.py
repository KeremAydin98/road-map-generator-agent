import requests  
import re
import json

'''url = 'http://localhost:5000/invoke_graph'  
input_data = {  
    "topic": "Kubernetes",  
    "level": "Junior",  
    "learning_style": "Reading",  
}  '''


url = 'http://localhost:5000/create_quiz'
input_data = {
    "description": "Deep dive into Angular's component architecture. Explore advanced data binding techniques",
    "learningType": "Writing",
    "level": "Beginner",
}
  
response = requests.post(url, json=input_data)  

print(response.json())