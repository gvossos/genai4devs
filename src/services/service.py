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

# pip install arize-phoenix
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

# Turn on instrumentation for OpenAI
# pip install openinference-instrumentation-openai openai
from phoenix.trace.openai import OpenAIInstrumentor


# Initialize Langchain auto-instrumentation
# pip install openinference-instrumentation-langchain
from phoenix.trace.langchain import LangChainInstrumentor


# Instrument multi agent applications using CrewAI
# pip install openinference-instrumentation-crewai
from openinference.instrumentation.crewai import CrewAIInstrumentor
# Initialize the CrewAIInstrumentor before your application code.

load_dotenv()

# Add API key - HOST ONLY
#OTEL_EXPORTER_OTLP_HEADERS = os.environ['OTEL_EXPORTER_OTLP_HEADERS'] # Your API Key here
#PHOENIX_CLIENT_HEADERS = os.environ['PHOENIX_CLIENT_HEADERS'] 
#PHOENIX_COLLECTOR_ENDPOINT = os.environ['PHOENIX_CLIENT_HEADERS']

# Phoenix Arize Project Name
#PHOENIX_PROJECT_NAME=os.environ['PHOENIX_PROJECT_NAME']

tracer_provider = trace_sdk.TracerProvider()
span_exporter = OTLPSpanExporter("http://localhost:6006/v1/traces")
span_processor = SimpleSpanProcessor(span_exporter)
tracer_provider.add_span_processor(span_processor)
trace_api.set_tracer_provider(tracer_provider)

OpenAIInstrumentor().instrument()
LangChainInstrumentor().instrument()
CrewAIInstrumentor().instrument()

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
        
    print(f"\nBegin -> Kick-off workflow...")
    
    workflow = GVProcess(json_str,pdf_upload, pdf_watchlist)
    result = workflow.compile_run_graph()
    
    final_result = result
    analysis_status = "GVProcess Workflow Completed"
    
    print(f"\nEnd -> Kick-off workflow...")
    
    return analysis_status, final_result

class MicroService():
    
    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist:str):
        
        print(f"\n    MicroService - initializing....")
        
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
    
    # Function to analyse the watchlist
    def perform_monitoring(self, status_state):
        
        print(f"\n        Begin -> MicroService - perform_monitoring....")
        
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
        
        print(f"\n        End -> MicroService - perform_monitoring....")
        
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
    perform_monitoring_flag: bool # SIMPLE genAI4Dev Use Case
    analyse_market_conditions_flag: bool # COMPLEX genAI4Dev Use Case
    analyse_transactions_flag: bool # COMPLEX genAI4Dev Use Case
    compare_switches_flag: bool # COMPLEX genAI4Dev Use Case
    execute_lstm_flag: bool
    finish_flag: bool
    config_output: Optional[str] = None
    perform_monitoring_output: Optional[str] = None
    analyse_market_conditions_output: Optional[str] = None
    analyse_transactions_output: Optional[str] = None
    compare_switches_output: Optional[str] = None
    execute_lstm_output: Optional[str] = None
    finish_output: Optional[str] = None
    
# Define the workflow process(== Graph)
class GVProcess():

    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist):
        
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
        
        # The graph will call MicroServices to perform business logic
        self.mService = MicroService(json_str, pdf_upload, pdf_watchlist)
        self.graph_builder = StateGraph(GraphState)
        
        #Step 4: define the nodes we will cycle between
          
        self.graph_builder.add_node("config", self.config_node)
        self.graph_builder.add_node("perform_monitoring", self.perform_monitoring_node)
        self.graph_builder.add_node("analyse_market_conditions", self.analyse_market_conditions_node)
        self.graph_builder.add_node("analyse_transactions", self.analyse_transactions_node)
        self.graph_builder.add_node("compare_switches", self.compare_switches_node)
        self.graph_builder.add_node("execute_lstm", self.execute_lstm_node)
        self.graph_builder.add_node("finish", self.finish_node)
        
        
        #Step 5: Set the entrypoint as 'config'
        self.graph_builder.set_entry_point("config")
        
        """
        # We now add conditional edges
        self.graph_builder.add_conditional_edges(
          # First, we define the node
          "config",
          # Next,, we pass in the function that will determine which node is called next
          self.config_next_node,
          # Finally, we pass in a mapping
          # The key(s) are strings, and the value(s) are other nodes
          # END is a special node marking that the graph should finish
          # What will happen is we will call 'perform_monitoring_next_node', and then the output of that
          # will be matched against the keys in the mapping.
          # Based on which one it matches, that node will be called.
          {
              "perform_monitoring": "perform_monitoring",
              "analyse_market_conditions": "analyse_market_conditions",
          }
      )
        
        self.graph_builder.add_conditional_edges(
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
                "analyse_transactions" : "analyse_transactions",
            }
        )
        
        self.graph_builder.add_conditional_edges(
            "analyse_market_conditions",
            self.analyse_market_conditions_next_node,
            {
                "analyse_transactions": "analyse_transactions",
                "compare_switches": "compare_switches",
            }
        )
        
        self.graph_builder.add_conditional_edges(
            "analyse_transactions",
            self.analyse_transactions_next_node,
            {
                "compare_switches": "compare_switches",
                "execute_lstm": "execute_lstm",
            }
        )
        
        self.graph_builder.add_conditional_edges(
            "compare_switches",
            self.compare_switches_next_node,
            {
                "execute_lstm": "execute_lstm",
                "finish": "finish",
            }
        )
        
        self.graph_builder.add_conditional_edges(
            "execute_lstm",
            self.execute_lstm_next_node,
            {
                "finish": "finish",
            }
        )
        
      """
      # We now add a normal edge between nodes
      # This means that after 'perform_monitoring' is called, 'analyse_market_conditions' node is called next
          
        self.graph_builder.add_edge('config', 'perform_monitoring')
        self.graph_builder.add_edge('perform_monitoring', 'analyse_market_conditions')
        self.graph_builder.add_edge('analyse_market_conditions', 'analyse_transactions')
        self.graph_builder.add_edge('analyse_transactions', 'compare_switches')
        self.graph_builder.add_edge('compare_switches', 'execute_lstm')
        self.graph_builder.add_edge('execute_lstm', 'finish')
        self.graph_builder.add_edge('finish', END)
        

    # Define Nodes in the Graph
    # We define nodes as functions

    # The agent: responsible for deciding what (if any) actions to take
    def config_node(self, state):
      print(f"\n    Begin -> configure state...")
      
      state['perform_monitoring_flag'] = True
      state['analyse_market_conditions_flag'] = False
      state['analyse_transactions_flag'] = False
      state['compare_switches_flag'] = False
      state['execute_lstm_flag'] = False
      state['finish_flag'] = False
      state['config_output'] = "I've performed Configuration"
      
      return state
    
    def perform_monitoring_node(self, state):
      print(f"\n    Begin -> perform_monitoring...")
      
      if state.get('perform_monitoring_flag'):
        _crewAI_response = self.mService.perform_monitoring(status_state="Waiting...")
        _perform_monitoring = "I've performed Monitoring"
      else:
         _perform_monitoring = "Not Activated"
      
      print(f"\n        {_perform_monitoring}")
      
      return {"perform_monitoring_output": _perform_monitoring}

    def analyse_market_conditions_node(self, state):
      print(f"\n    Begin -> analyse_market_conditions...")
      
      if state.get('analyse_market_conditions_flag'):
        _crewAI_response = self.mService.analyse_market_conditions(status_state="Waiting...")
        _analyse_market_conditions = "I've Analysed Market Conditions"
      else:
        _analyse_market_conditions = "Not Activated"
      
      print(f"\n        {_analyse_market_conditions}")
      
      return {"analyse_market_conditions_output": _analyse_market_conditions}
      
    def analyse_transactions_node(self, state):
      print(f"\n    Begin -> analyse_transactions...")
      
      if state.get('analyse_transactions_flag'):
        _crewAI_response = self.mService.analyse_transactions(status_state="Waiting...")
        _analyse_transactions = "I've Analysed Transactions"
      else:
        _analyse_transactions = "Not Activated"
      
      print(f"\n        {_analyse_transactions}")
      
      return {"analyse_transactions_output": _analyse_transactions}

    def compare_switches_node(self, state):
      print(f"\n    Begin -> compare_switches...")
      
      if state.get('compare_switches_flag'):
        _crewAI_response = self.mService.compare_switches(status_state="Waiting...")
        _compare_switches = "I've Compared Switches"
      else:
          _compare_switches = "Not Activated"
      
      print(f"\n        {_compare_switches}")
      
      return {"compare_switches_output": _compare_switches}

    def execute_lstm_node(self, state):
      print(f"\n    Begin -> execute_lstm...")
      
      if state.get('execute_lstm_flag'):
          _crewAI_response = self.mService.execute_lstm(status_state="Waiting...")
          _execute_lstm = "I've Executed LSTM"
      else:
          _execute_lstm = "Not Activated"
      
      print(f"\n        {_execute_lstm}")
      
      return {"execute_lstm_output": _execute_lstm}

    
    def finish_node(self, state):
      print("\n    Begin -> Finishing...")
      print(f"        >config_output: {state.get('config_output', '').strip()}")
      print(f"        >perform_monitoring_output: {state.get('perform_monitoring_output', '').strip()}")
      print(f"        >analyse_market_conditions_output: {state.get('analyse_market_conditions_output', '').strip()}")
      print(f"        >analyse_transactions_output: {state.get('analyse_transactions_output', '').strip()}")
      print(f"        >compare_switches_output: {state.get('compare_switches_output', '').strip()}")
      print(f"        >execute_lstm_output: {state.get('execute_lstm_output', '').strip()}")
      
      _finish = "I've finished"
      
      print(f"\n        End -> finish")
      return {"finish_output": _finish}

    """
    # Setup the CONDITIONAL EDGE functions that will determine which node is called next
    def config_next_node(self, state):
      #if state.get('config_output') != None:
      if state.get('perform_monitoring_flag'):
        return "perform_monitoring" 
      else:
        return "analyse_market_conditions"
    
    def perform_monitoring_next_node(self, state):
      #if state.get('perform_monitoring_output') != None: 
      if state.get('analyse_market_conditions_flag'):
        return "analyse_market_conditions" 
      else:
        return "analyse_transactions"

    def analyse_market_conditions_next_node(self, state):
      #if state.get('analyse_market_conditions_output') != None: 
      if state.get('analyse_transactions_flag'): 
        return "analyse_transactions"
      else:
        return "compare_switches"
        

    def analyse_transactions_next_node(self, state):
      #if state.get('analyse_transactions_output') != None:
      if state.get('compare_switches_flag'):
        return "compare_switches" 
      else:
        return "execute_lstm"

    def compare_switches_next_node(self, state):
        #if state.get('compare_switches_output') != None:
        if state.get('execute_lstm_flag'):
          return "execute_lstm" 
        else:
          return "finish"
        
    def execute_lstm_next_node(self, state):
        return "finish" 
    """
    
     # Step 3: Define the graph

    def compile_run_graph(self):
      # Step 6: Compile and Run the Graph
      # Finally, we compile our graph and run it with some input.

      print(f"\n\nGVProcess...executing")
      graph = self.graph_builder.compile()

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
       
      result = graph.invoke({'config_output': 'Kick-Off'}
                            )
      print(f"\n Workflow execution complete.  result={result}")
      
      return result
