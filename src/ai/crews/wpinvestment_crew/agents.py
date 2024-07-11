import os

#crewai-0.30.11
from crewai import Agent

from langchain_openai import ChatOpenAI as OpenAI
#from langchain_groq import ChatGroq

from dotenv import load_dotenv

from src.ai.tools.gv_rag_tools import GVRagTools
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

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
  #temperature=os.environ['LLM_MODEL_TEMP'],
  temperature=0,# Sets the creativity of the AI's responses. Lower values (closer to 0) make the AI more deterministic and less random.
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
## This class constructs Investment crew agents with a specific role, goals, backstory, and a set of tools that it can use to perform relevant tasks.
##  The parameters like verbose, allow_delegation, memory, max_iter, and max_rpm are used to configure the agent's behavior.  Refer to Agent parameters
#
#======================================================================================================================================================
class WPInvestmentCrewAgents():
  
  #-----------------------------
  ## The investment_analyst agent is designed to support financial advisors by conducting investment analysis using the provided tools.
  #----------------------------
  def investment_analyst(self):
    return Agent(
      role='Investment Analyst',
      goal=f"Use the tools you have to search and analyse investments on behalf of Financial Advisors.",
      backstory="""You help Financial Advisors with their research. Lets say an advisor wants to write a report justifying buy/sell decisions across a client portfolio. 
               In this case you will search and analyse the merits of the buy/sell decisions. You will not write the report, but conduct the analysis
               You will use the Search tools you have to conduct the analysis.
               
               """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm,
      tools=[
        GVRagTools.retrieve_json,
        GVRagTools.assess_impact_of_market_conditions,
        GVRagTools.search_online_pdf,
      ],
    )

  #------------------------------
  ## The performance_analyst agent is designed to lookup a URL and report on the absolute and relative performance of an investment using the provided tools. 
  #------------------------------
  def performance_analyst(self):
    return Agent(
      role='Performance Analyst',
      goal=f"""Lookup a URL and reports on the absolute and relative performance of an investment. 
                Use your tools to access the URL and provide a recommendation as to whether the investment has shown good 7 Day, 30 Day, 90 Day and Current FY (23-24) absolute and relative performance.
                Make sure to state the percentages in your response for both absolute and relative returns. 
            """,
      backstory="""
          You get your input from the URL and then output your findings based strictly on the performance returns from the URL.
          """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm,
      tools=[
        GVRagTools.performance_analysis_post,
      ]
    )
  
  #------------------------------
  ## The predictive_analyst agent is designed to predict the future value of the investment using the provided tools. 
  #------------------------------
  def predictive_analyst(self):
    return Agent(
      role='Predictive Analyst',
      goal=f"""Perform predictive price analysis of an investment. 
                Use your tools to access the price value time-series on an investment and provide a prediction of the future value of the investment 7 days into the future.
                Make sure to state the confidence band of the prediction. 
            """,
      backstory="""
          You get your input from the price value time-series and then output your prediction based strictly on the time-series data.
          """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm,
      tools=[
        GVRagTools.predictive_prophet,
      ]
    )
  
  #----------------------------
  ## The sentiment_analyst agent is designed to assess the market sentiment of an investment by looking through recent articles on the investment 
  ##  and decide if positive or negative market sentiment using the provided tools.
  #----------------------------
  def sentiment_analyst(self):
    return Agent(
      role='Sentiment Analyst',
      goal=f"""You help Financial Advisors assess the market sentiment of an investment by looking through recent articles on the investment and decide if positive or negative market sentiment. 
                You can use the tools you have to check for negative or positive market sentiment relating to the investment. 
            """,
      backstory="""
          Derive the market sentiment of the investment by searching the internet for the latest news using the tools you have access to.
          """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm,
      tools=[
        GVRagTools.search_news,
        GVRagTools.search_internet,
        #YahooFinanceNewsTool(),
        ],     
    )
  
  #------------------------
  ## The report_writer agent is designed to write an investment analysis report justifying buy/sell decisions across a client portfolio using the provided tools. 
  ##  It gets its input from the Investment Analyst, the Performance Analyst and the Sentiment Analyst agents 
  #------------------------
  def report_writer(self):
      return Agent(
        role='Report Writer',
        goal=f"You help Financial Advisors write an Investment Analysis report justifying buy/sell decisions across a client portfolio.",
        backstory="""
          You get your input from the Investment Analyst, the Performance Analyst and the Sentiment Analyst agents.
        """,
        verbose=agent_verbose_mode,
        allow_delegation=allow_delegation_param,
        memory=memory_param,
        max_iter=max_iter_param,
        max_rpm=max_rpm_param,
        llm=llm,
      )
   
  #-----------------------
  ## The trade_analyst agent is designed to write a recommendation report justifying whether to buy the investment using the provided tools. 
  ##  It gets its input from the from the Report Writer agent
  #-----------------------
  def trade_analyst(self):
    return Agent(
      role='Trading Analyst',
      goal=f"""You look through the written report and recommend whether to HOLD, BUY or SELL the investment, with a detailed explanation for your decision.
      """,
      backstory="""
          You get your input from the [report_writer_dashboard_task, predictive_analyst_task] and the market_data_tool and then write a file to the local filesystem.
          """,
      verbose=agent_verbose_mode,
      allow_delegation=allow_delegation_param,
      memory=memory_param,
      max_iter=max_iter_param,
      max_rpm=max_rpm_param,
      llm=llm, 
      tools=[
        GVRagTools.market_data_tool,
      ],
    )
    
  
