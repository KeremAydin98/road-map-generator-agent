import requests  
import re
import json

url = 'http://localhost:5000/invoke_graph'  
input_data = {  
    "topic": "Kubernetes",  
    "level": "Junior",  
    "learning_style": "Reading",  
}  
  
response = requests.post(url, json=input_data)  

print(response.json())