import requests  
  
url = 'http://localhost:5000/invoke_graph'  
input_data = {  
    "topic": "Kubernetes",  
    "level": "Junior",  
    "learning_style": "Reading",  
    "time_frame": "3 weeks",  
    "schedule_type": "Weekly"  
}  
  
response = requests.post(url, json=input_data)  
print(response.json())  