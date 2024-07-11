
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, RedirectResponse
import requests


from src.models.model import WPPayload, TradingViewPayload, crewAIPayload
from src.services.service import MicroService

import os
from typing import Optional
import json

# initialize Globals

# Define the subdirectory name
subdirectory = "src/repository"

analysis_status = {}  # Global dictionary to store analysis status

router = APIRouter()

# FastAPI Endpoints
#===================

@router.get("/")
async def index():
    #return JSONResponse(
    #    content={
    #        "message": "Hello from GV's tradingview demo!"
    #    }
    #)
    response = RedirectResponse(url='/gradio')
    return response
    #gr_url = os.environ['DEPLOYED_SERVICE'] + f"/gradio"
    #status_response = requests.get(gr_url)
    #if status_response.status_code != 200:
    #    return "Error checking status", "No result"
        