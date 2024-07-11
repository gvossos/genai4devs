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
## This class defines the Market Conditions crew tasks.  It makes specific reference to expected parameters that are required by the tools used to perform the tasks.
## Note that the parameter expected_output is MANDATORY, and async_execution indicates whether the task can be performed asynchronously.  Refer to Agent parameters
#
#===================================================================================================================================================================
class WPMarketCrewTasks():
  
  def activity_callback(self, output: TaskOutput):
    # Do something after the task is completed
    print(f"""\n
        Task completed! - {output.description}
    """)
  
  def market_data_analyst_activity(self, agent):
    return Task(
      description=dedent(f"""
                You are a financial expert and have a great knowledge about predicting the movement of the stock market after considering the impact of major macroeconomic data. 
                Macroeconomic indicators to be considered are:
                  - gdp_growth_annual - GDP growth (annual %).
                  - unemployment_total_percent - Unemployment total (% of labor force).
                  - consumer_price_index - Consumer Price Index
                  - inflation_consumer_prices_annual - Inflation, consumer prices (annual %)

                Provide trend insights on the economy from your analysis across last 3 quarters.
                Make sure to support your findings by referencing actual performance values.
                  
                Agent Tool parameters are:
                - No parameters are passed to the provided tools.
                
            """),
      expected_output='Trend insights on the economy from your analysis across last 12 months, 6 months and 3 months.',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"Market_conditions.txt"),
      callback=self.activity_callback,
    )    
  
