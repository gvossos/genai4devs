from pydantic import BaseModel, Field
from typing import Optional

# Define the Pydantic models that represents the structure of the Solution
class WPPayload(BaseModel):
    project_prompt_input: str
    mstarid_input: str
    apir_input: str
    fund_input: str
    
class TradingViewPayload(BaseModel):
    order_action: str
    order_contracts: int
    ticker: str
    strategy_position_size: int

class crewAIPayload(BaseModel):
    crew_project_prompt: str = Field(...,description='task')
    fund: str = Field(...,description='fund name')
    end_point: str = Field(..., description='API end-point')
    pdf1: Optional[str] = Field(..., description='File-path to Lonsec pdf')
    pdf2: Optional[str] = Field(..., description='File-path fo Market Conditions pdf')

