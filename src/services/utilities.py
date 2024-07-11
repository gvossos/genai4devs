# Function to start ngrok tunnel
import os
import uvicorn
from pyngrok import ngrok

def start_ngrok():
    ngrok.set_auth_token(os.environ['NGROK_API_KEY'])
    url = ngrok.connect(8000)
    print(f"ngrok tunnel \"http\" exposed at: {url}")

# Define a function to run FastAPI with uvicorn
def run_fastapi(app):
    uvicorn.run(app, workers=1, host="127.0.0.1", port=8000)


