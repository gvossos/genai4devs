import os

from crewai import Process, Crew

from dotenv import load_dotenv

from src.ai.crews.wpinvestment_crew.agents import WPInvestmentCrewAgents
from src.ai.crews.wpinvestment_crew.tasks import WPInvestmentCrewTasks


#================================================================================================
#
## This class defines and runs the Investment crew.
#
#================================================================================================
class WPInvestmentCrew:
  
  #-------------------------------------
  ## Initialize parameter values
  #-------------------------------------
  def __init__(self, crewAI_data):
    
    self._mstarid=crewAI_data['mstarid']
    self._apir=crewAI_data['apir']
    self._investment = crewAI_data['investment']
    self._benchmark=crewAI_data['benchmark']
    self._market_conditions = crewAI_data['market_conditions']
    
  #-----------------------------
  ## Define & Run the Investment crew Agent tasks
  #-----------------------------
  def run(self):
    
    ## Construct endpoints dynamically for the investment
    mstar_key = os.environ['MSTAR_KEY']
    
    investment_general_endpoint = f"https://api.morningstar.com/v2/service/mf/bnhvxx46zbbikql8/mstarid/{self._mstarid}?format=json&accesscode={mstar_key}"
    investment_strategy_endpoint = f"https://api.morningstar.com/v2/service/mf/InvestmentCriteria/mstarid/{self._mstarid}?format=json&accesscode={mstar_key}"
    investment_MStar_endpoint = f"https://api.morningstar.com/v2/service/mf/MorningstarRating/mstarid/{self._mstarid}?format=json&accesscode={mstar_key}"
    investment_weightings_endpoint = f"https://api.morningstar.com/v2/service/mf/Top10HoldingsV2/mstarid/{self._mstarid}?format=json&accesscode={mstar_key}"
    investment_redemption_endpoint = f"https://northonline.amp.com.au/NorthOnline/ViewDoc.aspx?documenttype=1&Type=APIR&Code={self._apir}"

    print(f"\n                General endpoint - {investment_general_endpoint}")
    print(f"                Strategy endpoint - {investment_strategy_endpoint}")
    print(f"                MStar endpoint - {investment_MStar_endpoint}")
    print(f"                Weightings endpoint - {investment_weightings_endpoint}")
    print(f"                Redemption endpoint - {investment_redemption_endpoint}")
    print(f"                Market Conditions filepath - {self._market_conditions}\n")
    
    #-------------------------------
    ## Define the Investment crew Agents & Tasks
    #-------------------------------
    agents = WPInvestmentCrewAgents()
    tasks = WPInvestmentCrewTasks()

    #----------------------------------------
    ## Instantiate the Investment crew agents
    #----------------------------------------
    investment_analyst_agent = agents.investment_analyst() 
    performance_analyst_agent = agents.performance_analyst()
    predictive_analyst_agent = agents.predictive_analyst()
    sentiment_analyst_agent = agents.sentiment_analyst()
    report_writer_agent = agents.report_writer()
    trade_analyst_agent = agents.trade_analyst()

    #---------------------------------------
    ## Instantiate the Investment crew tasks
    #---------------------------------------
    
    ## Instantiate the investment_general_task
    investment_general_task = tasks.investment_general_activity(
                                                    investment_analyst_agent,
                                                    investment_general_endpoint)
    ## Instantiate the investment_strategy_task
    investment_strategy_task = tasks.investment_strategy_activity(
                                                    investment_analyst_agent,
                                                    investment_strategy_endpoint)
    ## Instantiate the investment_MStar_task
    investment_MStar_task = tasks.investment_MStar_activity(
                                                    investment_analyst_agent,
                                                    investment_MStar_endpoint)
    
    ## Instantiate the investment_weightings_task
    investment_weightings_task = tasks.investment_weightings_activity(
                                                    investment_analyst_agent,
                                                    investment_weightings_endpoint)
    
    
    ## Instantiate the impact_of_market_conditions_task
    impact_of_market_conditions_task = tasks.impact_of_market_conditions_activity(
                                                          investment_analyst_agent,
                                                          self._market_conditions,
                                                          )
    
    
    ## Instantiate the investment_redemption_task
    investment_redemption_task = tasks.investment_redemption_activity(
                                                    investment_analyst_agent,
                                                    self._investment,
                                                    investment_redemption_endpoint
                                                    )
    
    
    ## Instantiate the performance_analyst_task
    performance_analyst_task = tasks.performance_analyst_activity(
                                                    performance_analyst_agent,
                                                    self._investment,
                                                    self._mstarid,
                                                    self._benchmark
                                                    )
    
    ## Instantiate the sentiment_analyst_task
    sentiment_analyst_task = tasks.sentiment_analyst_activity(
                                                    sentiment_analyst_agent,
                                                    self._investment
                                                    )  
    
    ## Instantiate the report_writer_dashboard_task based on the output context of all above tasks
    report_writer_dashboard_task = tasks.report_writer_dashboard_activity(
                                                    report_writer_agent,
                                                    self._investment
                                                    )
  
    report_writer_dashboard_task.context=[
                                investment_general_task,
                                investment_strategy_task,
                                investment_MStar_task,
                                investment_weightings_task,
                                impact_of_market_conditions_task,
                                investment_redemption_task,
                                performance_analyst_task,
                                sentiment_analyst_task,
                                ]
    
     ## Instantiate the predictive_analyst_task
    predictive_analyst_task = tasks.predictive_analyst_activity(
                                                    predictive_analyst_agent,
                                                    self._investment,
                                                    self._mstarid,
                                                    self._benchmark
                                                    )
    
    ## Instantiate the trade_analyst_task based on the output context of [report_writer_dashboard_task, predictive_analyst_task, economic_data_analyst_task]
    trade_analyst_task = tasks.trade_analyst_activity(
                                                    trade_analyst_agent,
                                                    self._investment
                                                    )
    
   
    trade_analyst_task.context=[
                                report_writer_dashboard_task,
                                ]
    
   
    
    #-------------------------------
    ## Define the multi-agent crew
    #-------------------------------
    investment_crew = Crew(
      agents=[
        investment_analyst_agent,
        performance_analyst_agent,
        sentiment_analyst_agent,
        report_writer_agent,
        predictive_analyst_agent,
        trade_analyst_agent,
      ],
      tasks=[
        investment_general_task,
        investment_strategy_task,
        investment_MStar_task,
        investment_weightings_task,
        impact_of_market_conditions_task,
        investment_redemption_task,
        performance_analyst_task,
        sentiment_analyst_task,
        report_writer_dashboard_task,
        predictive_analyst_task,
        trade_analyst_task,
              ],
      process=Process.sequential,
      #verbose=2, #1
      #full_output=True,
    )

    ## Start the crew's task execution
    result = investment_crew.kickoff()
    
    # After the crew execution, you can access the usage_metrics attribute to view the language model (LLM) usage metrics for all tasks executed by the crew. 
    # This provides insights into operational efficiency and areas for improvement.
    print(f"\n    {investment_crew.usage_metrics}")
    
    return(result)
  

