
import os

#crewai-0.30.11
from crewai import Agent

from langchain_openai import ChatOpenAI as OpenAI
#from langchain_groq import ChatGroq

from dotenv import load_dotenv

from src.ai.tools.gv_rag_tools import GVRagTools

# Load env variables
load_dotenv()

"""
llm = ChatGroq(
            temperature=os.environ['LLM_MODEL_TEMP'], 
            groq_api_key = os.environ['GROQ_API_KEY'], 
            model_name=os.environ['GROQ_LLM_MODEL']
        )
"""

llm = OpenAI(
  model=os.environ['LLM_MODEL'],
  temperature=os.environ['LLM_MODEL_TEMP'], # Sets the creativity of the AI's responses. Lower values (closer to 0) make the AI more deterministic and less random.
  openai_api_key=os.environ['OPENAI_API_KEY'],
)

#GLOBALS

# Define the relative subdirectory name
#  Use double backslashes to avoid them being interpreted as escape characters:
subdirectory = '.\\src\\repository'

#Agent parameters
agent_verbose_mode=False # Enables detailed logging of the agent's execution for debugging or monitoring purposes when set to True. Default is False
max_iter_param=10 # The maximum number of iterations the agent can perform before being forced to give its best answer. Default is 15
max_rpm_param=20 # The maximum number of requests per minute the agent can perform to avoid rate limits. It's optional and can be left unspecified, with a default value of None
memory_param=True # Indicates whether the agent should have memory or not, with a default value of False. This impacts the agent's ability to remember past interactions. Default is False
allow_delegation_param=False # Agents can delegate tasks or questions to one another, ensuring that each task is handled by the most suitable agent. Default is True

#======================================================================================================================================================
#
## This class constructs Market crew agents with a specific role, goals, backstory, and a set of tools that it can use to perform relevant tasks.
##  The parameters like verbose, allow_delegation, memory, max_iter, and max_rpm are used to configure the agent's behavior.  Refer to Agent parameters
#
#======================================================================================================================================================
class WPMarketCrewAgents():
  
  #----------------------------
  ## The market_data_analyst agent is designed to summarize the major macroeconomic data using the provided tool.
  #----------------------------
  def market_data_analyst(self):
    return Agent(
      role='Financial Market Data Analyst',
      goal=f"""You help Financial Advisors summarize the major financial market data using the provided tool.""",
      backstory="""
          Summarize the major financial market data.
          """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm,
      tools=[
        GVRagTools.economic_conditions,
        ],     
    )
  
  
  
  