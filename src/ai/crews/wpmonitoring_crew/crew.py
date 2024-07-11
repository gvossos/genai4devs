import os

from crewai import Process, Crew


from dotenv import load_dotenv

from src.ai.crews.wpmonitoring_crew.agents import WPMonitoringCrewAgents
from src.ai.crews.wpmonitoring_crew.tasks import WPMonitoringCrewTasks


# Load env variables
load_dotenv()

#================================================================================================
#
## This class defines and runs the Investment crew.
#
#================================================================================================
class WPMonitoringCrew:
  
  #-------------------------------------
  ## Initialize parameter values
  #-------------------------------------
  def __init__(self, crewAI_data):
    
    self._watchlist = crewAI_data['watchlist']
    self._asset_allocation_view = crewAI_data['asset_allocation_view']
    
  #-----------------------------
  ## Define & Run the Investment crew Agent tasks
  #-----------------------------
  def run(self):
    
    #-------------------------------
    ## Define the Investment crew Agents & Tasks
    #-------------------------------
    agents = WPMonitoringCrewAgents()
    tasks = WPMonitoringCrewTasks()

    #----------------------------------------
    ## Instantiate the Investment crew agents
    #----------------------------------------
    performance_analyst_agent = agents.performance_analyst() 

    #---------------------------------------
    ## Instantiate the Investment crew tasks
    #---------------------------------------
    
    ## Instantiate the watchlist_task
    watchlist_task = tasks.watchlist_activity(
                                  performance_analyst_agent,
                                  self._watchlist,
                                  )
    
    ## Instantiate the asset_allocation_view_task
    asset_allocation_view_task = tasks.asset_allocation_view_activity(
                                  performance_analyst_agent,
                                  self._asset_allocation_view,
                                  )
    
    
    #-------------------------------
    ## Define the multi-agent crew
    #-------------------------------
    investment_crew = Crew(
        agents=[
          performance_analyst_agent,
          ],
        tasks=[
          watchlist_task,
          asset_allocation_view_task,
        ],
        process=Process.sequential,
        memory=True,
    )

    ## Start the crew's task execution
    result = investment_crew.kickoff()
    
    # After the crew execution, you can access the usage_metrics attribute to view the language model (LLM) usage metrics for all tasks executed by the crew. 
    # This provides insights into operational efficiency and areas for improvement.
    print(f"\n    {investment_crew.usage_metrics}")
    
    return(result)
