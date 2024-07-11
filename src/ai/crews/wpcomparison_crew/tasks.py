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
## This class defines the Comparison crew tasks.  It makes specific reference to expected parameters that are required by the tools used to perform the tasks.
## Note that the parameter expected_output is MANDATORY, and async_execution indicates whether the task can be performed asynchronously.  Refer to Agent parameters
#
#===================================================================================================================================================================
class WPComparisonCrewTasks():
   
  def activity_callback(self, output: TaskOutput):
    # Do something after the task is completed
    print(f"""\n
        Task completed! - {output.description}
    """)
    
     
  #-----------------------------------------------------------------
  ## The comparison_analyst_activity task is designed to make a detailed comparison of the SELL and BUY investments based on the output context of report_writer_dashboard and the economic_data_analyst using the provided tools.
  ##  BUY Criteria are explicitly stated in the task description.
  ##  Agent tool parameters are the SELL and BUY investments 
  #------------------------------------------------------------------
  def investment_comparison_analyst_activity(
                          self, 
                          agent, 
                          sell_investment:str,
                          buy_investment: str,
                          ):
    return Task(
      description=dedent(f"""
                Provide a high level comparison of {sell_investment} and {buy_investment}.           
                
                Your comparison needs to compare the 2 investments across the following criteria.  The higher the value, the better - 
                  1. Absolute amd relative performance over 7d, 30d and 90d (the higher the better)
                  2. Net Asset Value 
                  3. Positive sentiment over the last 3 months.
                  4. Redemption not exceeding 30 days
                  5. Overall star rating 
                  6. Overall exposure to developed countries
                  7. Positive 7 day future value prediction
                  8. Positive 30 day future value prediction
                  9. Positive 60 day future value prediction
                  10. Positive 90 day future value prediction 
                  12. Favorable Economic analysis for the last Quarter
                
                Use bullet-points and state reasons supporting.
                Also, make sure to state whether you support the decision to sell {sell_investment} and buy {buy_investment}
                
                Agent Tool parameters are:
                - sell_investment: {sell_investment}
                - buy_investment: {buy_investment}
            """),
      expected_output='High level comparison',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"{sell_investment}_{buy_investment}_comparison.txt"),
      callback=self.activity_callback,
    )
  
  #-------------------------------------------------------
  ## The text_2_speech_activity task is designed to convert text to voice translation using the provided tools.
  ##  Agent tool parameter is the text file output generated from the comparison_analyst_task.
  #-------------------------------------------------------
  def text_2_speech_activity(
                          self, 
                          agent, 
                          sell_investment:str,
                          buy_investment:str,
                          ):
    return Task(
      description=dedent(f"""
                Translate the contents of the text file to voice file .mp3.
                
                Agent Tool parameters are:
                    - sell_investment: {sell_investment}
                    - buy_investment: {buy_investment}
            """),
      expected_output='Text to voice translation as an .mp3 file',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )
  
  #---------------------------------------------------------------
  ## The send_email_with_attachment_activity task is designed to send an email with the voice attachment from the [text_2_speech_activity] task using the provided tools.
  ##  Agent tool parameters are the Google SMTP server settings.
  #---------------------------------------------------------------
  def send_email_with_attachment_activity(
                        self, 
                        agent, 
                        sell_investment:str,
                        buy_investment:str,
                        ):
    return Task(
      description=dedent(f"""
                Send email with the .mp3 attachment generated from the text_2_speech agent.
                
                Agent Tool parameters are:
                - sell_investment: {sell_investment}
                - buy_investment: {buy_investment}
            """),
      expected_output='send email with .mp3 attachment',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )
  