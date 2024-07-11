import os

from crewai import Task
from crewai.tasks.task_output import TaskOutput

from textwrap import dedent
from dotenv import load_dotenv


# Load env variables
load_dotenv()

#Task parameters
async_execution_param=False # Indicates whether the task should be executed asynchronously, allowing the crew to continue with the next task without waiting for completion.

# Define the relative subdirectory name
#  Use double backslashes to avoid them being interpreted as escape characters:
subdirectory = '.\\src\\repository'


#===================================================================================================================================================================
#
## This class defines the Investment crew tasks.  It makes specific reference to expected parameters that are required by the tools used to perform the tasks.
## Note that the parameter expected_output is MANDATORY, and async_execution indicates whether the task can be performed asynchronously.  Refer to Agent parameters
#
#===================================================================================================================================================================
class WPMonitoringCrewTasks():
   
  def activity_callback(self, output: TaskOutput):
    # Do something after the task is completed
    print(f"""\n
        Task completed! - {output.description}
    """)
    
    
  #--------------------------------------------------------------------------------------------
  ## The impact_of_market_conditions_activity is an asynchronous task designed to assess impact of Market conditions of n top sectors / country exposures using the provided tools.
  ##  Agent tool parameter is the filepath to the PDF document summarizing the current market conditions
  #--------------------------------------------------------------------------------------------
  def watchlist_activity(self, agent, watchlist_pdf_file_path: str):
    return Task(
      description=dedent(f"""
                Write a detailed summary of investment performance over 30 days,60 days and since inception.  
                
                Use the following criteria:
                  1. benchmark vs holding  30 days, 60 days and since inception is positive.  Make sure to state performance values.
                  2. attribution performance over 30 days, 60 days ,and since inception is positive .  Make sure to state performance value deltas.
          
                Make sure to profile each of the investment holdings by referencing key attributes:
                - component
                - component type
                - Months held
                - Allocation
                - Factor Summary
                - Style
                - Sector
                
              Agent Tool parameters are:
                - watchlist_pdf_file_path: {watchlist_pdf_file_path}
                - query: assess the performance of investments over 30 days, 60 days and since inception.
                
            """),
      expected_output='Detailed summary of investment performance',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"1_monitoring_performance.txt"),
      callback=self.activity_callback,
    )
   
  
  #--------------------------------------------------------------------------------------------
  ## The impact_of_market_conditions_activity is an asynchronous task designed to assess impact of Market conditions of n top sectors / country exposures using the provided tools.
  ##  Agent tool parameter is the filepath to the PDF document summarizing the current market conditions
  #--------------------------------------------------------------------------------------------
  def asset_allocation_view_activity(self, agent, asset_allocation_view_pdf_file_path: str):
    return Task(
      description=dedent(f"""
                Provide a rationale justifying the poor performance of investments. 
                Include insights from the current asset allocation view document {asset_allocation_view_pdf_file_path} which provides commentary of the various market conditions and their impact on different asset classes & sectors.
                Do not use any other sources for this analysis.

              Agent Tool parameters are:
                - asset_allocation_view_pdf_file_path: {asset_allocation_view_pdf_file_path}
                - query: How do the current asset allocation views impact the performance of the holding investments?
                
            """),
      expected_output='Executive summary justifying the poor performance of each holding investment',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"2_monitoring_exec_summary.txt"),
      callback=self.activity_callback,
    )
   
