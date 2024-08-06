import os
import json

from src.ai.crews.wpcomparison_crew.crew import WPComparisonCrew
from src.ai.crews.wpinvestment_crew.crew import WPInvestmentCrew
from src.ai.crews.wpmarket_crew.crew import WPMarketCrew
from src.ai.crews.wpmonitoring_crew.crew import WPMonitoringCrew
from src.ai.ml.wplstm import WPLSTM, WPLSTMAttention

# pip install python-dotenv
from dotenv import load_dotenv

from IPython.display import Image, display

#import agentops

import phoenix as px

# Install OpenTelemetry
# OpenTelemetetry Protocol (or OTLP for short) is the means by which traces arrive from your application to the Phoenix collector. Phoenix currently supports OTLP over HTTP.
# pip install opentelemetry-api opentelemetry-instrumentation opentelemetry-semantic-conventions opentelemetry-exporter-otlp-proto-http
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from phoenix.evals import (
    HallucinationEvaluator,
    OpenAIModel,
    QAEvaluator,
    RelevanceEvaluator,
    run_evals,
)

from phoenix.session.evaluation import get_qa_with_reference, get_retrieved_documents
from phoenix.trace import DocumentEvaluations, SpanEvaluations

tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter("http://localhost:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)
trace_api.set_tracer_provider(tracer_provider)

# Turn on instrumentation for OpenAI
# pip install openinference-instrumentation-openai openai
from phoenix.trace.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument()

# Initialize Langchain auto-instrumentation
# pip install openinference-instrumentation-langchain
from phoenix.trace.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()

# Instrument multi agent applications using CrewAI
# pip install openinference-instrumentation-crewai
from openinference.instrumentation.crewai import CrewAIInstrumentor
# Initialize the CrewAIInstrumentor before your application code.
CrewAIInstrumentor().instrument()

load_dotenv()

# Add API key
OTEL_EXPORTER_OTLP_HEADERS = os.environ['OTEL_EXPORTER_OTLP_HEADERS'] # Your API Key here
PHOENIX_CLIENT_HEADERS = os.environ['PHOENIX_CLIENT_HEADERS'] 
PHOENIX_COLLECTOR_ENDPOINT = os.environ['PHOENIX_CLIENT_HEADERS']

# Phoenix Arize Project Name
PHOENIX_PROJECT_NAME=os.environ['PHOENIX_PROJECT_NAME']

# To view traces in Phoenix, you will first have to start a Phoenix server. You can do this by running the following:
session = px.launch_app()

# https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141 
# pip install langgraph
from langgraph.graph import StateGraph, END

from typing import Dict, TypedDict, Optional


# Micro-services
# api_routes point HERE for all business logic
# Make sure to import crews, lstm etc

#=================================================================
#
## MAIN ENTRY POINT from gradio UI
#
#=================================================================

def kickoff_workflow(json_str, pdf_upload, pdf_watchlist, status_state):
        
    print(f"\nKick-off workflow...")
    
    workflow = GVProcess(json_str,pdf_upload, pdf_watchlist)
    result = workflow.compile_run_graph()
    
    final_result = result
    analysis_status = "GVProcess Workflow Completed"
    
    return analysis_status, final_result

class MicroService():
    
    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist:str):
        
        print(f"\n    MicroService - initializing....")
        
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
    
    # Function to analyse the watchlist
    def perform_monitoring(self, status_state):
        
        print(f"\n        MicroService - perform_monitoring commencing....")
        
        #agentops.init(os.environ['AGENTOPS_API_KEY'])
        
        status_state = "Monitoring..."
        
        # Construct and run the WPMonitoringCrew
        crewAI_data = {
            "watchlist" : self._pdf_watchlist,
            "asset_allocation_view": self._pdf_upload,
        }
        
        print(f"\n        Run WPMonitoringCrew(crewAI_data)....")
        wp_crew = WPMonitoringCrew(crewAI_data)
        result = wp_crew.run()
        
        status_state = "Complete"
        
        #agentops.end_session("Success") # Success|Fail|Indeterminate
        
        return status_state, result

    # Function to analyse the Market Conditions
    def analyse_market_conditions(self, status_state):
        
        print(f"\n        MicroService - analyse_market_conditions commencing....")
        
        #agentops.init(os.environ['AGENTOPS_API_KEY'])
        
        status_state = "Analyzing..."
        
        print(f"\n        Run WPMarketCrew()....")
        wp_crew = WPMarketCrew()
        result = wp_crew.run()
        
        status_state = "Completed"
        
        
        #agentops.end_session("Success") # Success|Fail|Indeterminate
         
        # Return the new status and the final result string
        # These will be displayed in the 'status_output' and 'result_output' textboxes respectively
        return status_state, result

    # Function to analyse an investment
    def analyse_investment(self, type, transaction, pdf_upload):
        
        print(f"\n                MicroService - analyse_investment commencing....")
        
        #agentops.init(os.environ['AGENTOPS_API_KEY'], skip_auto_end_session=True)
        
        investment = transaction[type]
        
        #print(f"\nTransaction Type {type}:")
        print(f"              {type} Investment: {investment['Investment']}")
        
        # Construct and run the analysis task
        crewAI_data = {
            "mstarid": investment.get('MStarID', ''),
            "apir": investment.get('APIR', ''),
            "investment": investment.get('Investment', ''),
            "benchmark": investment.get('Benchmark', ''),
            "market_conditions": pdf_upload,
        }

        print(f"\n                Analysis for {type} Investment: {investment['Investment']} commencing....")
        
        print(f"\n                            Run WPInvestmentCrew(crewAI_data)....")
        wp_crew = WPInvestmentCrew(crewAI_data)
        result = wp_crew.run()
        
        print(f"\n                Analysis for {type} Investment: {investment['Investment']} completed.\n")
        
        #agentops.end_session("Success") # Success|Fail|Indeterminate
        
        return f"Analysis for {type} Investment: {investment['Investment']} completed."

    # Function to analyse the Portfolio switches (==transactions)
    def analyse_transactions(self, status_state):
        
        print(f"\n        MicroService - analyse_transactions commencing....")
         
        # Load the JSON object from the string directly
        transactions = json.loads(self._json_str)['transactions']

        results = []
        status_state = "Analyzing..."

        for i, transaction in enumerate(transactions, start=1):
            print(f"\n          ===========================================================================")
            print(f"            Transaction# {i}\n")
            results.append(self.analyse_investment("SELL", transaction, self._pdf_upload))
            results.append(self.analyse_investment("BUY", transaction, self._pdf_upload))
            
            #compare_switch(transaction,i)
            
            print(f"\n          ===========================================================================")
        
        final_result = "\n\n".join(results)
        status_state = "Completed"
         
        return status_state, final_result

    # Function to compare the sell and buy investments
    def compare_signal(self, transaction, i):
        
        print(f"\n            MicroService - compare_signal commencing....")
        
        #agentops.init(os.environ['AGENTOPS_API_KEY'])
        
        sell_investment_name = transaction["SELL"]['Investment']
        buy_investment_name = transaction["BUY"]['Investment']
        
        # Construct and run the analysis task
        crewAI_data = {
            "sell_investment": sell_investment_name,
            "buy_investment": buy_investment_name,
        }
        
        print(f"\n            --------------------------------------------------------------------------------")
        print(f"\n            Comparison of transaction# {i}: {sell_investment_name} and {buy_investment_name} commencing....")
        
        print(f"\n        Run WPComparisonCrew(crewAI_data)....")
        wp_crew = WPComparisonCrew(crewAI_data)
        result = wp_crew.run()
        
        print(f"\n            --------------------------------------------------------------------------------")
      
        #agentops.end_session("Success") # Success|Fail|Indeterminate
        
        return f"Performance comparison of switch {i} completed."
        
    def compare_switches(self, status_state):
            
            print(f"\n        MicroService - Compare Switches....")
            
            # Load the JSON object from the string directly
            transactions = json.loads(self._json_str)['transactions']

            results = []
            status_state = "Comparing..."

            for i, transaction in enumerate(transactions, start=1):
                print(f"\n        ===========================================================================")
                print(f"        Switch# {i}\n")
                results.append(self.compare_signal(transaction, i))
                
                #compare_switch(transaction,i)
                
                print(f"    \n===========================================================================")
            
            final_result = "\n\n".join(results)
            status_state = "Completed"

            return status_state, final_result

    # Function to analyse the Portfolio switches (==transactions)
    def analyse_transactions_fastrack(self, json_str, pdf_upload, status_state):
        
        print(f"\n    MicroService - analyse_transactions_fastrack commencing....")
        # Load the JSON object from the string directly
        transactions = json.loads(json_str)['transactions']

        results = []
        status_state = "Analyzing..."

        for i, transaction in enumerate(transactions, start=1):
            
            self.compare_switch(transaction,i)
            
        
        final_result = "\n\n".join(results)
        status_state = "Completed"

        # Return the new status and the final result string
        # These will be displayed in the 'status_output' and 'result_output' textboxes respectively
        return status_state, final_result

    def execute_lstm(self, status_state):
        
        print(f"\n    MicroService - run_lstm commencing....")
        status_state = "Analyzing..."
        
        #instantiate WPLSTM
        print(f"\n        Call Non-crewAI code....")
        wp_lstm=WPLSTM()
        
        final_result = wp_lstm.run()
        
        status_state = "Completed"
        final_result = f"Predicted next close price is: {final_result}"
        
        return status_state, final_result
    
    def wpLstm_attention_part_1(self, status_state):

        print(f"\n    MicroService - wpLstm_attention commencing....")
        
        status_state = "Analyzing..."
        
        wp_lstm_attention=WPLSTMAttention()
        
        final_result = wp_lstm_attention.run()

        status_state = "Completed"
        
        return status_state, final_result
        return final_result


############################################
# Langgraph
# Step 1: Define the Graph State
# First, we define the state structure for our graph.  We track states the following 6 states for the graph

class GraphState(TypedDict):
    start: Optional[str] = None
    perform_monitoring_output: Optional[str] = None
    analyse_market_conditions_output: Optional[str] = None
    analyse_transactions_output: Optional[str] = None
    compare_switch_output: Optional[str] = None
    execute_lstm_output: Optional[str] = None
    finished_output: Optional[str] = None
    
# Define the workflow process(== Graph)
class GVProcess():

    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist):
        
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
        
        # The graph will call MicroServices to perform business logic
        self.mService = MicroService(json_str, pdf_upload, pdf_watchlist)
        self.workflow = StateGraph(GraphState)
        
        ###################################################################
        # Workflow State (==Graph) switch configuration
        
        #self._start_flag = True
        self._perform_monitoring_flag = True # SIMPLE genAI4Dev Use Case
        self._analyse_market_conditions_flag = False # COMPLEX genAI4Dev Use Case
        self._analyse_transactions_flag = False # COMPLEX genAI4Dev Use Case
        self._compare_switches_flag = False # COMPLEX genAI4Dev Use Case
        self._execute_lstm_flag = False
        
        ##################################################################
        
        #Step 4: define the nodes we will cycle between
          
        #self.workflow.add_node("start", self.start_node)
        self.workflow.add_node("perform_monitoring", self.perform_monitoring_node)
        self.workflow.add_node("analyse_market_conditions", self.analyse_market_conditions_node)
        self.workflow.add_node("analyse_transactions", self.analyse_transactions_node)
        self.workflow.add_node("compare_switches", self.compare_switches_node)
        self.workflow.add_node("execute_lstm", self.execute_lstm_node)
        self.workflow.add_node("finish", self.finish_node)
        
        
        #Step 5: Set the entrypoint as 'start'
        self.workflow.set_entry_point("perform_monitoring")
        
        # Step 7: We now add conditional edges
        """
        self.workflow.add_conditional_edges(
            # First, we define the node
            "start",
            #Next,, we pass in the function that will determine which node is called next
            self.start_next_node,
            # Finally, we pass in a mapping
            # The kye(s) are strings, and the value(s) are other nodes
            # END is a special node marking that the graph should finish
            # What will happen is we will call 'start_next_node', and then the output of that
            # will be matched against the keys in the mapping.
            # Based on which one it matches, that node will be called.
            {
                "perform_monitoring": "perform_monitoring",
            }
        )
        """
        
        self.workflow.add_conditional_edges(
            # First, we define the node
            "perform_monitoring",
            # Next,, we pass in the function that will determine which node is called next
            self.perform_monitoring_next_node,
            # Finally, we pass in a mapping
            # The key(s) are strings, and the value(s) are other nodes
            # END is a special node marking that the graph should finish
            # What will happen is we will call 'perform_monitoring_next_node', and then the output of that
            # will be matched against the keys in the mapping.
            # Based on which one it matches, that node will be called.
            {
                "analyse_market_conditions": "analyse_market_conditions",
            }
        )
        
        self.workflow.add_conditional_edges(
            "analyse_market_conditions",
            self.analyse_market_conditions_next_node,
            {
                "analyse_transactions": "analyse_transactions",
            }
        )
        
        self.workflow.add_conditional_edges(
            "analyse_transactions",
            self.analyse_transactions_next_node,
            {
                "compare_switches": "compare_switches",
            }
        )
        
        self.workflow.add_conditional_edges(
            "compare_switches",
            self.compare_switches_next_node,
            {
                "execute_lstm": "execute_lstm",
            }
        )
        
      # Step 8: We now add a normal edge between nodes
      # This means that after 'start' is called, 'monitoring_performed' node is called next
          
        #self.workflow.add_edge('start', 'perform_monitoring')
        self.workflow.add_edge('perform_monitoring', 'analyse_market_conditions')
        self.workflow.add_edge('analyse_market_conditions', 'analyse_transactions')
        self.workflow.add_edge('analyse_transactions', 'compare_switches')
        self.workflow.add_edge('compare_switches', 'execute_lstm')
        self.workflow.add_edge('execute_lstm', 'finish')
        self.workflow.add_edge('finish', END)
        

        
    # Step 2: Define Nodes in the Graph
    # We define nodes as functions

    # The agent: responsible for deciding what (if any) actions to take
    """
    def start_node(self, state):
      print(f"\n    GVProcess - 0. Start Node...")
      
      _crewAI_response = self.handle_start()
      
      if _crewAI_response == "Not activated":
        _start = "Not activated"
        print(f"\n        Not Activated")
      else:
        _start = "I've Started"
        print(f"\n        Successfully Processed")
      
      return {"start_output": _start}

    #  A function to invoke the tools based on CONFIGURATION SWITCH values
    def handle_start(self):
      
      if self._start_flag:
        result = "Kick-Off"
      else:
        result = "Not activated"
      
      return result
    
    """
    
    def perform_monitoring_node(self, state):
      print(f"\n    GVProcess - 1. perform_monitoring...")
      
      _crewAI_response = self.handle_perform_monitoring()
      #print(f"perform_monitoring response: {_crewAI_response}")
      
      if _crewAI_response == "Not activated":
        _perform_monitoring = "Not activated"
        print(f"\n        Not Activated")
      else:
        _perform_monitoring = "I've performed Monitoring"
        print(f"\n        Successfully Processed")
      
      return {"perform_monitoring_output": _perform_monitoring}

    #  A function to invoke the tools based on CONFIGURATION SWITCH values
    def handle_perform_monitoring(self):
      # print(f"    ....perform_monitoring by calling crewAI....")
      
      if self._perform_monitoring_flag:
        result = self.mService.perform_monitoring(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def analyse_market_conditions_node(self, state):
      print(f"\n    GVProcess - 2. analyse_market_conditions...")
      
      _crewAI_response = self.handle_analyse_market_conditions()
      # print(f"analyse_market_conditions response: {_crewAI_response}")
      if _crewAI_response == "Not activated":
        _analyse_market_conditions = "Not activated"
        print(f"\n        Not Activated")
      else:
        _analyse_market_conditions = "I've analysed market conditions"
        print(f"\n        Successfully Processed")
      
      return {"analyse_market_conditions_output": _analyse_market_conditions}

    def handle_analyse_market_conditions(self):
      #print(f"    ....handle_analyse_market_conditions by calling crewAI....")
      
      if self._analyse_market_conditions_flag :
        result = self.mService.analyse_market_conditions(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def analyse_transactions_node(self, state):
      print(f"\n    GVProcess - 3. analyse_transactions...")
      
      _crewAI_response = self.handle_analyse_transactions()
      # print(f"analyse_transactions response: {_crewAI_response}")
      if _crewAI_response == "Not activated":
        _analyse_transactions = "Not activated"
        print(f"\n        Not Activated")
      else:
        _analyse_transactions = "I've analysed transactions"
        print(f"\n        Successfully Processed")

      return {"analyse_transactions_output": _analyse_transactions}

    def handle_analyse_transactions(self):
      # print(f"    ....handle_analyse_transactions by calling crewAI....")

      if self._analyse_transactions_flag:
        result = self.mService.analyse_transactions(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def compare_switches_node(self, state):
      print(f"\n    GVProcess - 4. compare_switches...")
      
      _crewAI_response = self.handle_compare_switches()
      #print(f"compare_switch response: {_crewAI_response}")
      
      if _crewAI_response == "Not activated":
        _compare_switches = "Not activated"
        print(f"\n        Not Activated")
      else:
        _compare_switches = "I've performed Switch comparison"
        print(f"\n        Successfully Processed")
      
      return {"compare_switches_output": _compare_switches}

    def handle_compare_switches(self):
      # print(f"    ....compare_switch by calling crewAI....")
      
      if self._compare_switches_flag:
        result = self.mService.compare_switches(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def execute_lstm_node(self, state):
      print(f"\n    GVProcess - 5. execute_lstm...")
      
      _crewAI_response = self.handle_execute_lstm()
      # print(f"run_lstm response: {_crewAI_response}")
      
      if _crewAI_response == "Not activated":
        _execute_lstm = "Not activated"
        print(f"\n        Not Activated")
      else:
        _execute_lstm = "I've run LSTM"
        print(f"\n        Successfully Processed")
        
      return {"execute_lstm_output": _execute_lstm}

    def handle_execute_lstm(self):
      # print(f"....handle_run_lstm by calling crewAI....")
      
      if self._execute_lstm_flag:
        result = self.mService.execute_lstm(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def finish_node(self, state):
      print("\n    5. Finishing...")
      print(f"        >start_output: {state.get('start_output', '').strip()}")
      print(f"        >perform_monitoring_output: {state.get('perform_monitoring_output', '').strip()}")
      print(f"        >analyse_market_conditions_output: {state.get('analyse_market_conditions_output', '').strip()}")
      print(f"        >analyse_transactions_output: {state.get('analyse_transactions_output', '').strip()}")
      print(f"        >compare_switch_output: {state.get('compare_switch_output', '').strip()}")
      print(f"        >run_lstm_output: {state.get('run_lstm_output', '').strip()}")
      
      _finished = "I've finished"
      return {"finished_output": _finished}

    # Step 6: setup the CONDITIONAL EDGE functions that will determine which node is called next
    """
    def start_next_node(self, state):
      # print("        ...satisfies start_next_node rule")
      if state.get('perform_monitoring_output') != None :
        return "perform_monitoring" 
    """
    
    def perform_monitoring_next_node(self, state):
      # print("        ...satisfies monitoring_performed_next_node rule")
      if state.get('analyse_market_conditions_output') != None: 
        return "analyse_market_conditions" 

    def analyse_market_conditions_next_node(self, state):
      # print("        ...satisfies market_conditions_analysed_next_node rule")
      if state.get('analyse_transactions_output') != None: 
        return "analyse_transactions" 

    def analyse_transactions_next_node(self, state):
      # print("        ...satisfies transactions_analysed_next_node rule")
      if state.get('compare_switch_output') != None:
        return "compare_switches" 

    def compare_switches_next_node(self, state):
        # print("        ...satisfies switches_compared_next_node rule")
        if state.get('run_lstm_output') != None:
          return "execute_lstm" 
        else:
          return "start"
    
     # Step 3: Define the graph

    def compile_run_graph(self):
      # Step 6: Compile and Run the Graph
      # Finally, we compile our graph and run it with some input.

      print(f"\n\nGVProcess...executing")
      graph = self.workflow.compile()

      try:
          # Generate the image
          img = graph.get_graph().draw_mermaid_png()
          
          # Save the image to a file
          with open("graph_image.png", "wb") as file:
              file.write(img)
          
          # Display the image
          display(Image(img))

      except Exception as e:
          print(f"An error occurred: {e}")

      """  
      inputs = {
        "transactions": {self._json_str},
        "pdf_upload" : {self._pdf_upload}
        }
      """
      
      result = graph.invoke({'start':'Kick-off'})
      
      print(f"\n Workflow execution complete.  result={result}")
      
      return result
