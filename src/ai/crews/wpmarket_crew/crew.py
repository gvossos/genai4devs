import os

from crewai import Process, Crew


from dotenv import load_dotenv

from src.ai.crews.wpmarket_crew.agents import WPMarketCrewAgents
from src.ai.crews.wpmarket_crew.tasks import WPMarketCrewTasks


# Load env variables
load_dotenv()

#================================================================================================
#
## This class defines and runs the Market crew.
#
#================================================================================================
class WPMarketCrew:
  
  #-------------------------------------
  ## Initialize parameter values
  #-------------------------------------
  def __init__(self):
    
    #self._market_conditions = crewAI_data['market_conditions']
    pass
    
  #-----------------------------
  ## Define & Run the Investment crew Agent tasks
  #-----------------------------
  def run(self):
    
    #-------------------------------
    ## Define the Investment crew Agents & Tasks
    #-------------------------------
    agents = WPMarketCrewAgents()
    tasks = WPMarketCrewTasks()

    #----------------------------------------
    ## Instantiate the Investment crew agents
    #----------------------------------------

    market_data_analyst_agent = agents.market_data_analyst()

    #---------------------------------------
    ## Instantiate the Investment crew tasks
    #---------------------------------------
    
    
    ## Instantiate the economic_data_analyst_task
    market_data_analyst_task = tasks.market_data_analyst_activity(
                                                    market_data_analyst_agent,
                                                    )  
    
    #-------------------------------
    ## Define the multi-agent crew
    #-------------------------------
    market_crew = Crew(
        agents=[
          market_data_analyst_agent,
          ],
        tasks=[
          market_data_analyst_task,
        ],
        process=Process.sequential,
        #verbose=2, #1
        #full_output=True,
    )

    ## Start the crew's task execution
    result = market_crew.kickoff()
    
    # After the crew execution, you can access the usage_metrics attribute to view the language model (LLM) usage metrics for all tasks executed by the crew. 
    # This provides insights into operational efficiency and areas for improvement.
    print(f"\n    {market_crew.usage_metrics}")
    
    return(result)