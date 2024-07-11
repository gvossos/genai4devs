from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware

import gradio as gr

from dotenv import load_dotenv

from src.ui.gui import create_gradio_interface
from src.services.utilities import start_ngrok, run_fastapi  
from src.services.api_routes import router as api_routes

import logging

# initialize FastAPI
app = FastAPI()

# add Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include the routes from api_routes.py
app.include_router(api_routes)

# Load env variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)


# Mount a gradio.Blocks to an existing FastAPI application.
demo = create_gradio_interface()
gr.mount_gradio_app(app, demo, path="/gradio")


# If using Google App Engine (GAE) use  app.yaml instead
if __name__ == "__main__":
    # Start ngrok when running LOCAL.
    start_ngrok()

    # Start FastAPI when running LOCAL 
    run_fastapi(app)

