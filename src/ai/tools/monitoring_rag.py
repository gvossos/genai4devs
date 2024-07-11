import os
from datetime import datetime, timedelta

from pathlib import Path

from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.tools import tool

from PyPDF2 import PdfReader
from dotenv import load_dotenv


# pip install boto3
import boto3
from botocore.exceptions import ClientError

import logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Define the relative subdirectory name
#  Use double backslashes to avoid them being interpreted as escape characters:
subdirectory = '.\\src\\repository'


#======================================================================================================================================================
#
## This class constructs the RAG tools used by Agents to perform assigned tasks.
#
#======================================================================================================================================================
class MonitoringTools():

    #====================================
    ## Agent Tools: investment_analyst
    #====================================

    #---------------------------------
    ## The watchlist tool is designed to assess the performance of investments on the watchlist
    #---------------------------------
    @tool("assess the performance of investments on the watchlist")
    def assess_watchlist_performance(watchlist_pdf_file_path:str, query: str):
        """
        Asses the impact of market conditions on the investment.
        
        :param watchlist_pdf_file_path: file path to PDF document
        :param query: query prompt to use for assessing the impact of market conditions
        :return: response string
        
        """
    
        _path = watchlist_pdf_file_path
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
            answers = "watchlist_pdf_file_path is None"
        
        return answers
    
    #---------------------------------
    ## The assess_aa_view_impacts tool is designed to assess the impact of market conditions on the investment
    #---------------------------------
    @tool("assess the impact of asset allocation views on the investment")
    def assess_aa_view_impacts(asset_allocation_view_pdf_file_path:str, query: str):
        """
        Asses the impact of market conditions on the investment.
        
        :param asset_allocation_view_pdf_file_path: file path to PDF document
        :param query: query prompt to use for assessing the impact of market conditions
        :return: response string
        
        """
    
        _path = asset_allocation_view_pdf_file_path
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
            answers = "market_conditions_file_path is None"
        
        return answers
    
    ##utility functions NOT accessed by Agents
    
    

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
    



