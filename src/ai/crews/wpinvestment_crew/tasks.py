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
class WPInvestmentCrewTasks():
  
  #------------------------------------------------------
  ## Callback functions 
  #------------------------------------------------------
  #---------------------------------------------------------------
  ## The upload_to_s3_callback is designed to Upload file to S3 after the report_writer_dashboard_activity task is completed.
  ##  Parameters to the tool are the fund_name and report_writer_dashboard_activity task TaskOutput.
  #---------------------------------------------------------------
  def upload_to_s3_callback(self, output: TaskOutput):
    if output.raw_output is not None:
        #GVRagTools.upload_file_to_s3(fund_name,output.raw_output)
        print("upload_to_s3_callback - uploading file to AWS S3")
    else:
        print("upload_to_s3_callback failed to upload to AWS S3\n")
  
  
  def activity_callback(self, output: TaskOutput):
      # Do something after the task is completed
      print(f"""\n
          Task completed! - {output.description}
      """)

  #------------------------------------------------------------
  ## The investment_general_activity is an asynchronous task designed to retrieve general investment information from Morningstar using the provided tools.
  ##  Agent tool parameter is the REST API endpoint
  #------------------------------------------------------------
  def investment_general_activity(self, agent, investment_general_endpoint: str):
      return Task(
        description=dedent(
                  f"""
                  Parse the returned JSON from the REST API {investment_general_endpoint} and return the following fund information using the following keys.              
                  Access this from data -> api 
                    `FSCBI-CategoryName`
                    `Link`,
                    `FSCBI-APIR`
                    `FSCBI-Ticker`
                    `FNA-FundNetAssets`
                    `FSCBI-ProviderCompanyN`
                    `FB-PrimaryIndexId`
                    `F S C B I- Morningstar Index Id`
                    `C E- Country Exposure`
                    `A T- Leveraged Fund`
                    `A T- Index Fund`
                    `A T- Delayed Pricing`
                    `R E- Regional Exposure`
                                                
                  Agent Tool parameters are:
                  - url: {investment_general_endpoint}
                  
              """),
        expected_output='Output should be a bullet list of returned key-value pairs ',
        async_execution=async_execution_param,
        agent=agent,
        callback=self.activity_callback,
      )
  
  #-------------------------------------------------------------
  ## The investment_strategy_activity is an asynchronous task designed to retrieve investment strategy from Morningstar using the provided tools.
  ##  Agent tool parameter is the REST API endpoint
  #-------------------------------------------------------------
  def investment_strategy_activity(self, agent, investment_strategy_endpoint: str):
      return Task(
        description=dedent(
                  f"""
                  Parse the returned JSON from the REST API {investment_strategy_endpoint} and return the following fund information using the following keys.
                  Access this from data -> api 
                    `InvestmentStrategy`
                                  
                  Agent Tool parameters are:
                  - url: {investment_strategy_endpoint}
                  
              """),
        expected_output='Output should be a bullet list of returned key-value pairs ',
        async_execution=async_execution_param,
        agent=agent,
        callback=self.activity_callback,
      )
  
  #-----------------------------------------------------------
  ## The investment_MStar_activity is an asynchronous task designed to retrieve the investment rating from Morningstar using the provided tools.
  ##  Agent tool parameter is the REST API endpoint
  #------------------------------------------------------------
  def investment_MStar_activity(self, agent, investment_MStar_endpoint: str):
      return Task(
        description=dedent(
                  f"""
                  Parse the returned JSON from the REST API {investment_MStar_endpoint} and return the following fund information using the following keys.
                  Access this from data -> api 
                    `RatingOverall`
                                                
                  Agent Tool parameters are:
                  - url: {investment_MStar_endpoint}
                  
              """),
        expected_output='Output should be a bullet list of returned key-value pairs',
        async_execution=async_execution_param,
        agent=agent,
        callback=self.activity_callback,
      )
  
  #---------------------------------------------------------------
  ## The investment_weightings_activity is an asynchronous task designed to retrieve the investment Country & Sector Weightings from Morningstar using the provided tools.
  ##  Agent tool parameter is the REST API endpoint
  #---------------------------------------------------------------
  def investment_weightings_activity(self, agent, investment_weightings_endpoint: str):
    return Task(
      description=dedent(
                f"""
                Return the top 5 'Sector' exposures by 'Weighting', and separately top 5 'Country' exposures by 'Weighting'.
                Do this by parsing the returned JSON from the REST API {investment_weightings_endpoint}.
                Access this from data -> api -> HoldingDetail.
                
                For 'Sector' exposures by 'Weighting', use the 'Sector' attribute and corresponding 'Weighting' attribute to answer the question.
                  - Extract 'Weighting' value
                  - sum 'Weighting' for each 'Sector'
                  - sort 'Sector' by highest total 'Weighting' and select top 5.
                For 'Country' exposures by 'Weighting', use the 'Country' attribute and corresponding 'Weighting' attribute to answer the question.
                  - Extract 'Weighting' value
                  - sum 'Weighting' for each 'Country'
                  - sort 'Sector' by highest total 'Weighting' and select top 5.

                Agent Tool parameters are:
                - url: {investment_weightings_endpoint}
                
            """),
      expected_output='Output should be two bullet list summaries: One of the top 5 sector exposures (sector, weighting) and another of top 5 country exposures (country, weighting)',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )
  
  #--------------------------------------------------------------------------------------------
  ## The impact_of_market_conditions_activity is an asynchronous task designed to assess impact of Market conditions of top 5 sectors / country exposures using the provided tools.
  ##  Agent tool parameter is the filepath to the PDF document summarizing the current market conditions
  #--------------------------------------------------------------------------------------------
  def impact_of_market_conditions_activity(self, agent, market_conditions: str):
    return Task(
      description=dedent(f"""
                How do the current market conditions impact the portfolio sector and country exposures output from ["investment_weightings_task"]
                The market conditions document {market_conditions} provides a comprehensive analysis of various market conditions and their impact on different sectors.
                
                The market conditions file contains 5 rows corresponding to 5 key sectors (Equities, Credit, Property, Infrastructure and Government Bonds), and 7 columns:
                  - Activity: insight into key market factor impacts
                  - Inflation: insight into inflation impacts
                  - Monetary Conditions: insights into monetary condition impacts
                  - Flows & Liquidity: insight into investor trend impacts
                  - Supply & Corporate: insights into trends in the supply of securities on issue and corporate buyback impacts
                  - Valuation: insight into corporate valuation impacts
                  - Key Risks: impact of key risks 
                              
                Agent Tool parameters are:
                  - market_conditions_file_path: {market_conditions}
                  - query: How do the current market conditions impact the portfolio sector and country exposures?
                
            """),
      expected_output='A bullet list summary of the top 5 most important sector impacts and separately, a list summary of the top 5 most important country exposures.',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )
  
  #---------------------------------------------------------------
  ## The investment_redemption_activity is an asynchronous task designed to check the redemption conditions of the investment from the online Public Disclosure Statement (PDS) PDF using the provided tools.
  ##  Agent tool parameter is the REST API endpoint
  #----------------------------------------------------------------
  def investment_redemption_activity(self, agent, investment:str, investment_redemption_endpoint:str):
    return Task(
      description=dedent(f"""
                What is the investment's redemption policy?

                Agent Tool parameters are:
                - pdf_url: {investment_redemption_endpoint}
                - investment: {investment}
                - query: What is the investment's redemption policy?
                
            """),
      expected_output='A bullet list summary of the redemption policy',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )    
    
  #------------------------------------------------------
  ## The performance_analyst_activity is an asynchronous task designed to assess the absolute and relative to benchmark performance of the investment from internal system Performance URL using the provided tools.
  ##  Agent tool parameters are the investment name, investment id and the investment's benchmark id
  #------------------------------------------------------
  def performance_analyst_activity(
                          self, 
                          agent, 
                          investment:str,
                          mstarid:str,
                          benchmark:str
                          ):
    return Task(
      description=dedent(f"""
                Assess the absolute and relative performance of {investment} over 7,30,90,180 days and Current FY (23-24).
                Check if the performance of the investment  meets the following criteria - 
                  1. absolute performance over 7,30,90,180 days and Current FY (23-24) is positive.  Make sure to state performance values.
                  2. relative to benchmark performance over 7,30,90,180 days and FY (23-24) is positive .  Make sure to state performance value deltas.
                            
                Agent Tool parameters are:
                  - mstarid: {mstarid}
                  - benchmark: {benchmark}
            """),
      expected_output='Detailed investment performance findings',
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )
  
  #-------------------------------------------------------
  ## The sentiment_analyst_activity is an asynchronous task designed to assess the current market sentiment of the investment from various finance news sources using the provided tools.
  ##  Agent tool parameter is the investment name
  #-------------------------------------------------------
  def sentiment_analyst_activity(self, agent, investment:str):
    return Task(
      description=dedent(f"""
                What is the current market sentiment of {investment}.
                If no information is available, simply return 'Neutral market sentiment for {investment}.

                Agent Tool parameters are:
                - investment: {investment} 
                
            """),
      expected_output=f"summary of market sentiment for {investment}",
      async_execution=async_execution_param,
      agent=agent,
      callback=self.activity_callback,
    )    
  
  #--------------------------------------------------
  
  ## The report_writer_dashboard_activity task is designed to write an executive summary in Dashboard format of findings from the output context of all proceeding asynchronous tasks using the provided tools.
  ##  Agent tool parameters to the tool are the investment name 
  #--------------------------------------------------
  def report_writer_dashboard_activity(
                            self, 
                            agent, 
                            investment:str
                            ):
    return Task(
      description=dedent(f"""
                Write Detailed Trading Recommendation.
                Make sure to reference the impact of market conditions on the top 5 asset classes and include a summary of the performance analysis.
                
                Agent Tool parameters are:
                **Parameters**: 
                    - fund: {investment}
            """),
      expected_output='Detailed Trading Recommendation',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"{investment}_ES.txt"),
      callback=self.upload_to_s3_callback,
    )
  
  #------------------------------------------------------
  ## The predictive_analyst_activity is an asynchronous task designed to predict the future value of an investment using time-series price data using the provided tools.
  ##  Agent tool parameters are the investment name, investment id and the investment's benchmark id
  #------------------------------------------------------
  def predictive_analyst_activity(
                          self, 
                          agent, 
                          investment:str,
                          mstarid:str,
                          benchmark:str
                          ):
    return Task(
      description=dedent(f"""
                Summarize the predicted future value of {investment} over 30 days from today using FB Prophet
                Do not repeat any of the data, instead provide an insight into positive or negative movement, and expected magnitude.
                            
                Agent Tool parameters are:
                  - mstarid: {mstarid}
                  - benchmark: {benchmark}
                
            """),
      expected_output='Summary of future value predictions',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"{investment}_Prophet.txt"),
      callback=self.activity_callback,
    )
  
  #-------------------------------------------------------
  
  #-----------------------------------------------------------------
  ## The trade_analyst_activity task is designed to make BUY recommendation based on the output context of Report Writer Agent.
  ##  BUY Criteria are explicitly stated in the task description.
  ##  Agent tool parameter is the investment name 
  #------------------------------------------------------------------
  def trade_analyst_activity(
                          self, 
                          agent, 
                          investment:str
                          ):
    return Task(
      description=dedent(f"""
                
                Write a detailed recommendation justifying whether to BUY investment - {investment} using the BUY criteria below.
                
                  1. Positive absolute amd relative performance over 7d, 30d and 90d
                  2. Net Asset Value greater than $10,000,000
                  3. No negative sentiment on the fund last 3 months.
                  4. Redemption not exceeding 30 days
                  5. Overall rating greater than 2 stars
                  6. Overall exposure to developed countries greater than 60%
                  7. Positive or neutral market sentiment

                Make sure to support your findings by referencing the BUY criteria, including the actual investment performance values for each criteria.
                
                Agent Tool parameters are:
                  - fund: {investment}
            """),
      expected_output='Detailed trading recommendation.',
      async_execution=async_execution_param,
      agent=agent,
      output_file=os.path.join(subdirectory,f"{investment}_recommendation.txt"),
      callback=self.activity_callback,
    )
  

