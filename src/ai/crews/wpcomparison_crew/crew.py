
from crewai import Process, Crew

from src.ai.crews.wpcomparison_crew.agents import WPComparisonCrewAgents
from src.ai.crews.wpcomparison_crew.tasks import WPComparisonCrewTasks


#================================================================================================
#
## This class defines and runs the Comparison crew.
#
#================================================================================================
class WPComparisonCrew:
  
  #-------------------------------------
  ## Initialize parameter values
  #-------------------------------------
  def __init__(self, crewAI_data):
    
    self._sell_investment = crewAI_data['sell_investment']
    self._buy_investment = crewAI_data['buy_investment']
    
    
  #-----------------------------
  ## Define & Run the Comparison Analyst Agent tasks
  #-----------------------------
  def run(self):
    
    #-------------------------------
    ## Define the Comparison Analyst Agents & Tasks
    #-------------------------------
    agents = WPComparisonCrewAgents()
    tasks = WPComparisonCrewTasks()

    #-------------------------
    ## Instantiate the comparison agents
    #-------------------------
    investment_comparison_analyst_agent = agents.investment_comparison_analyst()
    text_2_speech_agent = agents.text_2_speech()
    email_agent = agents.email_with_attachment()

    #------------------------
    ## Instantiate the tasks
    #-------------------------
    
    ## Instantiate the comparison_report_writer_dashboard_task based on the output context of all above tasks
    investment_comparison_analyst_task = tasks.investment_comparison_analyst_activity(
                                                    investment_comparison_analyst_agent,
                                                    self._sell_investment,
                                                    self._buy_investment,
                                                    )
    
    ## Instantiate the text_2_speech_task based on the output context of [trade_analyst_task]
    text_2_speech_task = tasks.text_2_speech_activity(
                                                    text_2_speech_agent,
                                                    self._sell_investment,
                                                    self._buy_investment,
                                                    )
    
    ## Instantiate the send_email_with_attachment_task based on the voice translation output context of [text_2_speech-task]
    send_email_with_attachment_task = tasks.send_email_with_attachment_activity(
                                                    email_agent,
                                                    self._sell_investment,
                                                    self._buy_investment,
                                                    )
    
    #-------------------------------
    ## Define the multi-agent crew
    #-------------------------------
    comparison_crew = Crew(
        agents=[
          investment_comparison_analyst_agent,
          text_2_speech_agent,
          email_agent,
          ],
        tasks=[
          investment_comparison_analyst_task,
          text_2_speech_task,
          send_email_with_attachment_task,
        ],
        process=Process.sequential,
        #verbose=2, #1
        #full_output=True,
    )

    ## Start the crew's task execution
    result = comparison_crew.kickoff()
    
    # After the crew execution, you can access the usage_metrics attribute to view the language model (LLM) usage metrics for all tasks executed by the crew. 
    # This provides insights into operational efficiency and areas for improvement.
    print(f"\n    {comparison_crew.usage_metrics}")
    
    return(result)