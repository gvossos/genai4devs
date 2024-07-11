import os
from datetime import datetime, timedelta

""" Not required 
#pip install python-docx
from docx import Document
"""
from pathlib import Path

from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.tools import tool

from openai import OpenAI
client = OpenAI() # for text-to-audio

from PyPDF2 import PdfReader
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

# pip install boto3
import boto3
from botocore.exceptions import ClientError

import logging
logging.basicConfig(level=logging.INFO)

import re #regex

import json

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

#pip install eodhd -U
from eodhd import APIClient as eodhdClient

#pip install prophet -U
from prophet import Prophet

#pip install -U scikit-learn
from sklearn.preprocessing import LabelEncoder

import pandas as pd


# Load environment variables
load_dotenv()

# Define the relative subdirectory name
#  Use double backslashes to avoid them being interpreted as escape characters:
subdirectory = '.\\src\\repository'

    
client = OpenAI()

#======================================================================================================================================================
#
## This class constructs the RAG tools used by Agents to perform assigned tasks.
#
#======================================================================================================================================================
class GVRagTools():

  #====================================
  ## Agent Tools: economic_data_analyst
  #====================================

  #------------------------------
  ## The economic_conditions tool is designed to extract Market Conditions data from .txt file
  #------------------------------
  @tool("extract Market Conditions data")
  def market_data_tool():
    """
    Extract Financial market conditions data.
    
    """
    
    market_conditions_doc = os.path.join(subdirectory,f"Market_conditions.txt")
    
    # Extract market conditions data
    market_conditions_text = GVRagTools.__extract_text(market_conditions_doc)
    
    # Concatenate sell and buy performance texts
    market_performance = f"Market Conditions:\n{market_conditions_text}"
          
    return market_performance
  
  
  #====================================
  ## Agent Tools: comparison_analyst
  #====================================

  #------------------------------
  ## The comparison_tool is designed to extract investment performance data from txt file and then analyze and compare the investments and make a recommendation
  #------------------------------
  @tool("extract investment performance data and then analyze and compare the investments and make a recommendation")
  def comparison_tool(sell:str, buy:str):
    """
    Extract investment performance data from txt file and then analyze and compare the investments and make a recommendation.
    
    :param sell: name of the sell investment
    :param buy: name of the buy investment
    :return: response string
    
    """
    
    sell_investment_doc = os.path.join(subdirectory,f"{sell}_recommendation.txt")
    buy_investment_doc = os.path.join(subdirectory,f"{buy}_recommendation.txt")
    
    print(f"\n    You are in the comparison_tool about to compare {sell} vs {buy}")
    
    # Extract investment performance data
    sell_performance_text = GVRagTools.__extract_text(sell_investment_doc)
    buy_performance_text =  GVRagTools.__extract_text(buy_investment_doc)
    
    # Concatenate sell and buy performance texts
    concatenated_performance = f"Sell Investment Performance:\n{sell_performance_text}\n\nBuy Investment Performance:\n{buy_performance_text}"
          
    return concatenated_performance
  

  #====================================
  ## Agent Tools: investment_analyst
  #====================================

  #------------------------------
  ## The retrieve_json tool is designed to retrieve JSON data from a REST API using the GET method
  #------------------------------
  @tool("Retrieve JSON string from user specified URL")
  def retrieve_json(url:str):
    """
    Retrieve JSON data from a REST API endpoint using GET.
    
    :param url: REST API endpoint
    :return: JSON object
    
    """
    
    # Define the endpoint URL
    _url = url 
      
    headers = {
    'Content-Type': 'application/json'
    }

    # Send a GET request to the URL
    response = requests.get(_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        returned_json = response.json()
        print(returned_json)
        answer = returned_json
    else:
        print(f"retrieve_json: Failed to retrieve data. Status code: {response.status_code}")
        answer = "Failed"
      
    return answer
  
  #---------------------------------
  ## The assess_impact_of_market_conditions tool is designed to assess the impact of market conditions on the investment
  #---------------------------------
  @tool("assess the impact of market conditions on the investment")
  def assess_impact_of_market_conditions(market_conditions_file_path:str, query: str):
    """
    Asses the impact of market conditions on the investment.
    
    :param market_conditions_file_path: file path to PDF document
    :param query: query prompt to use for assessing the impact of market conditions
    :return: response string
    
    """
  
    _path = market_conditions_file_path
    _prompt = query
  
    if _path is not None:
      with open(_path, "rb") as pdf:
          pdf_reader = PdfReader(pdf)
          
          # Text variable will store the pdf text
          text = ""
          for page in pdf_reader.pages:
              text += page.extract_text()

      # Split the text into chunks using Langchain's CharacterTextSplitter
      text_splitter = CharacterTextSplitter(
              separator = "\n",
              chunk_size = 1000,
              chunk_overlap  = 150,
              length_function = len
          )
      
      docs = text_splitter.create_documents([text])
      retriever = FAISS.from_documents(
        docs, OpenAIEmbeddings()
      ).as_retriever()
      answers = retriever.get_relevant_documents(_prompt, top_k=1)
      answers = "\n\n".join([a.page_content for a in answers])
    else:
      answers = "pdf2 is None"
    
    return answers
  
  #---------------------------------
  ## The search_online_pdf tool is designed to search online PDF
  #---------------------------------
  @tool("Search online PDF")
  def search_online_pdf(pdf_url, investment, query):
    
    """
    Search an online PDF document for a given query
    
    :param pdf_url: file path to PDF document
    :param investment: investment name
    :param query: query prompt to use
    :return response string
    
    """
    
    initial_response = requests.get(pdf_url)
    if initial_response.status_code != 200:
        return "Failed to retrieve the initial PDF URL."

    pdf_url = GVRagTools.__find_pdf_url(initial_response.text)
    if not pdf_url:
        return "No PDF URL found in the HTML content."

    print(f"\nPDS embedded link is here - {pdf_url}\n")
    
    pdf_content = GVRagTools.__download_pdf(pdf_url)
    if not pdf_content:
        return "Failed to download the PDF."

    text = GVRagTools.__extract_text_from_pdf(investment, pdf_content)
    if not text:
      return "PDF file locked and can't be edited."

    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap  = 150,
            length_function = len
        )
      
    docs = text_splitter.create_documents([text])
    retriever = FAISS.from_documents(
      docs, OpenAIEmbeddings()
    ).as_retriever()
    answers = retriever.get_relevant_documents(query, top_k=1)
    answers = "\n\n".join([a.page_content for a in answers])
    
    return answers
    
  #====================================
  ## Agent Tools: performance_analyst
  #====================================
  #-------------------------------------------------
  ## The performance_analysis_post tool is designed to derive the absolute and relative performance of an investment using the POST method
  #-------------------------------------------------
  @tool("conduct performance analysis via post")
  def performance_analysis_post(mstarid:str, benchmark:str):
    """
    Determine the absolute and relative performance of an investment
    
    :param mstarid: MStarId of the investment
    :param benchmark: MStarId of the investments's benchmark
    :return response string
    
    """
    
    # Create a session object
    s = requests.Session()

    wp_usr = os.environ['WP_USER']
    wp_pwd = os.environ['WP_PWD']
    
    credentials = (wp_usr, wp_pwd)

    # Set the URL
    url = 'https://l.wealthpilot.com.au/compareInvestments/index'

    identifiers=f"{mstarid},{benchmark}"
    
    # Set the parameters as a dictionary

    params={
      "identifier":identifiers,
      "pricing":"LAST_PRICE",
      "indexType":"NTR",
      "startDate":"",
      "endDate":"",
      "sort":"DEFAULT",
      "scoringModel":"BALANCED",
      "_alignStart":"",
      "alignStart":"on",
      "_alignEnd":"",
      "_includeTables":"",
      "_includeRelative":"",
      "_monthlyTimePeriod":"",
      "_dailyPrice":"",
      "dailyPrice":"on",
      "controller":"compareInvestments",
      "action":"index"
      }
    
    # Make the POST request with parameters and authentication
    response = s.post(url, auth=credentials, data=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the table with id="theTable"
        table = soup.find('table', id='measures')
        
        # Check if the table was found
        if table:
            print("Table found:")
            
            # Find all the rows in the table
            rows = table.find_all('tr')
            
            # Iterate over each row
            for row in rows:
                # Find all the cells within the row (including header cells)
                cells = row.find_all(['td', 'th'])
                
                # Extract and print the text from each cell
                for cell in cells:
                    print(cell.get_text().strip(), end=' | ')
                print()  # Add a newline after each row for better readability
            
            response=table
            
        else:
            print("Table with id='theTable' was not found.")
            response="Table with id='theTable' was not found."
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")
        response=f"Failed to retrieve the page. Status code: {response.status_code}"
    
    return(response)
  
  #====================================
  ## Agent Tools: predictive_analyst
  #====================================
  #-------------------------------------------------
  ## The predictive_prophet tool is designed to predict the future 7 day price value of the investment using FB Prophet
  #-------------------------------------------------
  @tool("predict the future 7 day price value of the investment using FB Prophet")
  def predictive_prophet(mstarid:str, benchmark:str):
    """
    predict the future 7 day price value of the investment using time-series price data
    
    :param mstarid: MStarId of the investment
    :param benchmark: MStarId of the investments's benchmark
    :return response string
    
    """
    
    #api = APIClient(os.environ['EODHD_API_KEY'])
    #df = api.get_historical_data("GSPC.INDX", "d")
    
    mstar_key = os.environ['MSTAR_KEY']
    
    # Define the endpoint URL
    _start = "2023-04-01"
    _end ="2024-04-01"
    
    #Daily URL
    _url = f"https://api.morningstar.com/service/mf/DailyReturnIndex/mstarid/{mstarid}?format=json&accesscode={mstar_key}&&startdate={_start}&enddate={_end}&frequency=d"  
    print(f"predictive_prophet - _url= {_url}")
  
    headers = {
    'Content-Type': 'application/json'
    }

    # Send a GET request to the URL
    response = requests.get(_url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data  = response.text
        #print(returned_json)
        
        ## FB Prophet business logic here.......
        
        """
        In a Prophet forecast, several key columns are particularly important for interpreting the results:

          * yhat: This is the forecasted value. It represents the model’s prediction for the target variable at a specific time. It’s the primary outcome of interest as it provides the predicted value based on the model’s understanding of underlying trends and seasonalities.
          
          * yhat_lower and yhat_upper: These columns provide the lower and upper bounds of the forecast’s confidence interval, respectively. They offer a range within which the actual value is expected to fall, giving an idea of the prediction’s uncertainty. Wider intervals suggest greater uncertainty, while narrower intervals suggest more confidence in the prediction.
          
          * trend: This column shows the long-term trend component of the data. It indicates the general direction in which the data is moving, abstracted from seasonal fluctuations or other cyclical patterns.
          
          * ds: Although not a forecast output per se, the “ds” column is crucial as it contains the date-time stamps for each prediction, indicating when each forecasted value (yhat) applies.
        
        """
          
        try:
          
          # Parse the JSON data into a Python dictionary
          data = json.loads(json_data)
          
          # Access the 'r' attribute
          r_attribute = data['data']['api']['r']
          
          #print("Successfully parsed the JSON and accessed 'r' attribute.")
          
          # Convert this to a pandas DataFrame
          df = pd.DataFrame(r_attribute)

          # Convert 'date' column to datetime
          #df["d"] = pd.to_datetime(df["d"])

          # Rename the columns
          df_prophet = df[["d", "v"]].rename(columns={"d":"ds", "v":"y"})

          df["ds"] = df.index
          
          # Show the DataFrame
          #print(df_prophet)
          
          # Initialise and Fit the Model
          # We have our data ready in the correct format, and now we need to intialise and fit our model.
          
          model = Prophet(daily_seasonality=True)
          model.fit(df_prophet)
          
          # Define the periods for which you want to generate forecasts
          periods = [7, 30, 60, 90]

          # Create a dictionary to hold the forecast results for each period
          forecasts = {}

          for period in periods:
              # Create future DataFrame for the next 'period' days
              future = model.make_future_dataframe(periods=period)
              
              # Forecast the future values
              forecast = model.predict(future)
              
              # Extract the relevant forecast data
              forecasts[period] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail(period)

          # Access the forecasts for each period
          forecast_7_days = forecasts[7]
          forecast_30_days = forecasts[30]
          forecast_60_days = forecasts[60]
          forecast_90_days = forecasts[90]

          # Print or analyze the forecasts as needed
          #print("7-Day Forecast:\n", forecast_7_days)
          #print("\n30-Day Forecast:\n", forecast_30_days)
          #print("\n60-Day Forecast:\n", forecast_60_days)
          #print("\n90-Day Forecast:\n", forecast_90_days)
          
          #plot_7d = model.plot(forecast_7_days)
          
          answer = forecasts
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
        except KeyError as e:
            print(f"Key not found in JSON structure: {e}") 
    else:
        print(f"retrieve_json: Failed to retrieve data. Status code: {response.status_code}")
        answer = "Failed"
        
    return answer
    
  #---------------------------------------------------
  ## The economic_conditions tool is designed to assess the current major economic conditions using EODHD.
  #-------------------------------------------------
  @tool("assess the current major economic conditions using EODHD get_macro_indicators_data")
  def economic_conditions():
    """
    Assess the current major economic conditions
    
    """
    
    # https://eodhd.com/financial-apis/python-financial-libraries-and-code-samples/ 
    # https://eodhd.com/financial-apis/macroeconomics-data-and-macro-indicators-api/
    # https://eodhd.com/financial-academy/stocks-price-prediction-examples/forecast-economic-variables-with-arima/
    
    api = eodhdClient(os.environ['EODHD_API_KEY'])
  
    indicator_list = ["gdp_growth_annual", "unemployment_total_percent", "consumer_price_index", "inflation_consumer_prices_annual"]
    
    # An empty dictionary to store DataFrames
    dataframes = {}
    
    # Calculate the date 10 years ago from today
    ten_years_ago = datetime.now() - timedelta(days=10*365)
    
    for value in indicator_list:
      # Fetching the data and creating a DataFrame
      df = pd.DataFrame(api.get_macro_indicators_data("USA", value))[["Date", "Value"]]
      df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
      df = df.set_index("Date").sort_index()
      
      # Filter to keep only the last 10 years
      df_filtered = df[df.index >= ten_years_ago]
      
      # Set frequency to Quarterly
      df_filtered = df_filtered.asfreq(freq='Q')

      # Reset the index so 'Date' becomes a regular column
      df_reset = df.reset_index()

      # Storing the DataFrame in the dictionary with a unique key
      dataframes[value] = df_reset

    # Convert each DataFrame in `dataframes` to JSON, including the 'Date' column
    json_objects = {}
    for key, df in dataframes.items():
        # Convert the DataFrame to a JSON string
        # 'records' orientation includes both 'Date' and 'Value' in the JSON array of objects
        json_objects[key] = df.to_json(orient='records', date_format='iso')
        
    # Convert the entire `json_objects` dictionary to a JSON-formatted string
    json_string = json.dumps(json_objects, indent=4)

    #print(json_string)
    return json_string
    
  
  #======================================================================================================================================================
  ## Agent Tools: sentiment_analyst
  #======================================================================================================================================================

  #------------------------------
  ## The search_news tool is designed to search the internet about a a given investment and return relevant results using Google serper REST API
  #------------------------------
  @tool("Search investment news on the internet")
  def search_news(investment):
    """
    Search the internet for news about an investment using Google Serper and return relevant results
    
    :param investment: investment name
    :return  response string
    
    """
  
    top_result_to_return = 4
    url = "https://google.serper.dev/news"
    payload = json.dumps({"q": investment})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['news']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)
  
  #------------------------------
  ## The search_internet tool is designed to search the internet about a a given investment and return relevant results using Google serper REST API
  #------------------------------
  @tool("Search the internet")
  def search_internet(investment):
    """
    Search the internet for current information regarding an investment using Google Serper and return relevant results
    
    :param investment: investment name
    :return response string
    
    """
    
    top_result_to_return = 4
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": investment})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    string = []
    for result in results[:top_result_to_return]:
      try:
        string.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    return '\n'.join(string)
  
  #======================================================================================================================================================
  ## Agent Tools: report_writer
  #======================================================================================================================================================

  #------------------------------
  ## The write_to_subdirectory tool is designed to save a file to the local filesystem
  #------------------------------
  @tool("write file to local subdirectory")
  def write_to_subdirectory(filename, content):
    """
    Writes content to a file within the './src/repository' subdirectory.
    
    :param filename: The name of the file to write the content to.
    :param content: The content to write to the file.
    
    """
    
    # Define the subdirectory name
    subdirectory = './src/repository'
    
    # Ensure the subdirectory exists
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    
    # Construct the full path to the file within the subdirectory
    file_path = os.path.join(subdirectory, f"{filename}_dashboard.txt")
    
    # Open the file for writing and write the content
    with open(file_path, 'w') as file:
        file.write(content)
    
    print(f"Content written to {file_path}")
    
  #------------------------------
  ## The upload_file_to_s3 tool is designed to upload files to AWS S3
  #------------------------------
  @tool("Upload a file to an S3 bucket")
  def upload_file_to_s3(file_name, object_name=None):
    """
    Upload a file to an AWS S3 bucket

    :param file_name: File to upload
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """
    
    aws_bucket=os.environ['AWS_BUCKET']
    
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, aws_bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
  
  #======================================================================================================================================================
  ## Agent Tools: trade_analyst
  #======================================================================================================================================================
  # Refer to report_writer tools

  #======================================================================================================================================================
  ## Agent Tools: text_2_speech
  #======================================================================================================================================================

  #---------------------------------
  ## The convert_txt_2_speech tool is designed to open, read and convert a text if file into voice and save to mp3
  #---------------------------------
  @tool("convert text to speech")
  def convert_txt_2_speech(sell_investment:str, buy_investment:str):
    """
    Open, read and convert text file to voice and save to mp3.

    :param sell_investment: Sell investment name
    :param buy_investment: Buy investment name
    :return: response string
    
    """
    
    txt_file_path=os.path.join(subdirectory,f"{sell_investment}_{buy_investment}_comparison.txt")
    
    speech_file_name = f"{sell_investment}_{buy_investment}_speech.mp3"
    
    if os.path.exists(txt_file_path):
      
      # Open the .txt file for reading ('r' mode is default, but specified here for clarity)
      with open(txt_file_path, 'r', errors='ignore') as file:
          # Read the content of the file
          content = file.read()
          
      response = client.audio.speech.create(
      model="tts-1",
      voice="nova",
      input=content
      )

      response.stream_to_file(os.path.join(subdirectory, speech_file_name))
    
    else:
      print(f"\n    __extract_text_from_docx ERROR: The file does not exist. Check the file path: {txt_file_path}")
      return f"__extract_text_from_docx ERROR: The file does not exist. Check the file path: {txt_file_path}"
  

  #=====================================
  ## Agent Tools: email_with_attachment
  #=====================================
  #------------------------------
  ## The send_email_with_attachment2 tool is designed to send email with attachment using Google Mail
  #------------------------------
  @tool("send email with attachment")
  def send_email_with_attachment2(sell_investment:str, buy_investment:str):
    """
    Send an email, and include the .mp3 attachment
      
    :param sell_investment: Sell investment name
    :param buy_investment: Buy investment name
    :return: response string
    
    """
    
    smtp_server = os.environ['SMTP_SERVER']
    port=465
    sender_email = os.environ['SENDER_EMAIL']
    sender_password = os.environ['SENDER_PWD']
    receiver_email = os.environ['RECEIVER_EMAIL']
    subject=f"Switch recommendation for {sell_investment} and {buy_investment}"
    body=f"Switch recommendation for {sell_investment} and {buy_investment}"
    
                    
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    # Add body to the email
    message.attach(MIMEText(body, 'plain'))

    mp3_file_path=os.path.join(subdirectory,f"{sell_investment}_{buy_investment}_speech.mp3")
    
    if os.path.exists(mp3_file_path):
      
      # Open the file to be sent
      with open(mp3_file_path, 'rb') as attachment_file:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment_file.read())

      # Encode file in base64
      encoders.encode_base64(part)

      # Add header as key/value pair to attachment part
      part.add_header(
        'Content-Disposition',
        f'attachment; filename= {mp3_file_path}',
      )

      # Add attachment to message and convert message to string
      message.attach(part)
      text = message.as_string()

      # Connect to the server and send the email
      with smtplib.SMTP_SSL(smtp_server, port) as server:
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, text)
        
      return "Email sent successfully."

    else:
      print(f"\n    send_email_with_attachment2 ERROR: The file does not exist. Check the file path: {mp3_file_path}")
      return f"send_email_with_attachment2 ERROR: The file does not exist. Check the file path: {mp3_file_path}"


  ##utility functions NOT accessed by Agents
  
  #---------------------------------
  ## The read_content tool is designed to Read webpage content using BeautifulSoup
  #---------------------------------
  @tool("Read webpage content")
  def __read_content(url: str) -> str:
      """Read content from a webpage."""
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')
      text_content = soup.get_text()
      return text_content[:5000]

  #---------------------------------
  ## The download_pdf tool is designed to Download a PDF file from the given URL
  #---------------------------------
  def __download_pdf(pdf_url):
    """Download a PDF file from the given URL."""
    response = requests.get(pdf_url)
    if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
        return response.content
    return None

  #---------------------------------
  ## The find_pdf_url tool is designed to ind the first PDF URL in the given HTML content.
  #---------------------------------
  def __find_pdf_url(html_content):
    """Find the first PDF URL in the given HTML content."""
    pattern = 'https:\\/\\/doc.morningstar.com\\/Document\\/[^\"]+'
    match = re.search(pattern, html_content)
    return match.group() if match else None

  #---------------------------------
  ## The extract_text_from_pdf tool is designed to extract text from the given PDF content, limiting to a maximum number of pages..
  #---------------------------------
  def __extract_text_from_pdf(investment, pdf_content, max_pages=3):
    """Extract text from the given PDF content, limiting to a maximum number of pages."""
    
    # Define the subdirectory name
    subdirectory = './src/repository'
    
    # Ensure the subdirectory exists
    if not os.path.exists(subdirectory):
        os.makedirs(subdirectory)
    
    # Construct the full path to the file within the subdirectory
    file_path = os.path.join(subdirectory, f"{investment}_PDS.pdf")
    
    
    with open(file_path, 'wb') as pds_file:
        pds_file.write(pdf_content)
    pdf_reader = PdfReader(file_path)
    text = ""
    for i in range(min(len(pdf_reader.pages), max_pages)):
        text += pdf_reader.pages[i].extract_text() or ""
    return text
  
  #---------------------------------
  ## The extract_text_from_docs tool is designed to extract text from the given file, limiting to a maximum number of pages..
  #---------------------------------
  def __extract_text(investment):
    """Extracts text from a DOCX file."""
    
    if os.path.exists(investment):
      
      # Open the .txt file for reading ('r' mode is default, but specified here for clarity)
      with open(investment, 'r', errors='ignore') as file:
          # Read the content of the file
          content = file.read()
          
      return content
    
    else:
      print(f"\n    __extract_text_from_docx ERROR: The file does not exist. Check the file path: {investment}")
      return f"__extract_text_from_docx ERROR: The file does not exist. Check the file path: {investment}"
  


