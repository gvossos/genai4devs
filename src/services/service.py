import os
import json

from src.ai.crews.wpcomparison_crew.crew import WPComparisonCrew
from src.ai.crews.wpinvestment_crew.crew import WPInvestmentCrew
from src.ai.crews.wpmarket_crew.crew import WPMarketCrew
from src.ai.crews.wpmonitoring_crew.crew import WPMonitoringCrew
from src.ai.ml.wplstm import WPLSTM, WPLSTMAttention

# pip install python-dotenv
from dotenv import load_dotenv

import agentops

# https://medium.com/@cplog/introduction-to-langgraph-a-beginners-guide-14f9be027141 
# pip install langgraph
from langgraph.graph import StateGraph, END

from typing import Dict, TypedDict, Optional

load_dotenv()

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
    result = workflow.execute()
    
    final_result = result
    analysis_status = "GVProcess Workflow Completed"
    
    return analysis_status, final_result

class MicroService():
    
    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist:str):
        
        print(f"\n    MicroService - initializing....")
        
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
    
    # Function to analyze the watchlist
    def perform_monitoring(self, status_state):
        
        print(f"\n        MicroService - perform_monitoring commencing....")
        
        agentops.init(os.environ['AGENTOPS_API_KEY'])
        
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
        
        agentops.end_session("Success") # Success|Fail|Indeterminate
        
        return status_state, result

    # Function to analyze the Market Conditions
    def analyze_market_conditions(self, status_state):
        
        print(f"\n        MicroService - analyze_market_conditions commencing....")
        
        agentops.init(os.environ['AGENTOPS_API_KEY'])
        
        status_state = "Analyzing..."
        
        print(f"\n        Run WPMarketCrew()....")
        wp_crew = WPMarketCrew()
        result = wp_crew.run()
        
        status_state = "Completed"
        
        
        agentops.end_session("Success") # Success|Fail|Indeterminate
         
        # Return the new status and the final result string
        # These will be displayed in the 'status_output' and 'result_output' textboxes respectively
        return status_state, result

    # Function to analyze an investment
    def analyze_investment(self, type, transaction, pdf_upload):
        
        print(f"\n                MicroService - analyze_investment commencing....")
        
        agentops.init(os.environ['AGENTOPS_API_KEY'], skip_auto_end_session=True)
        
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

    # Function to analyze the Portfolio switches (==transactions)
    def analyze_transactions(self, status_state):
        
        print(f"\n        MicroService - analyze_transactions commencing....")
         
        # Load the JSON object from the string directly
        transactions = json.loads(self._json_str)['transactions']

        results = []
        status_state = "Analyzing..."

        for i, transaction in enumerate(transactions, start=1):
            print(f"\n          ===========================================================================")
            print(f"            Transaction# {i}\n")
            results.append(self.analyze_investment("SELL", transaction, self._pdf_upload))
            results.append(self.analyze_investment("BUY", transaction, self._pdf_upload))
            
            #compare_switch(transaction,i)
            
            print(f"\n          ===========================================================================")
        
        final_result = "\n\n".join(results)
        status_state = "Completed"
         
        return status_state, final_result

    # Function to compare the sell and buy investments
    def compare_signal(self, transaction, i):
        
        print(f"\n            MicroService - compare_signal commencing....")
        
        agentops.init(os.environ['AGENTOPS_API_KEY'])
        
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
      
        agentops.end_session("Success") # Success|Fail|Indeterminate
        
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

    # Function to analyze the Portfolio switches (==transactions)
    def analyze_transactions_fastrack(self, json_str, pdf_upload, status_state):
        
        print(f"\n    MicroService - analyze_transactions_fastrack commencing....")
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

    def run_lstm(self, status_state):
        
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
# First, we define the state structure for our graph. In this example, our state includes the user’s question, the classification of the question, and a response.

class GraphState(TypedDict):
    perform_monitoring_output: Optional[str] = None
    analyze_market_conditions_output: Optional[str] = None
    analyze_transactions_output: Optional[str] = None
    compare_switch_output: Optional[str] = None
    run_lstm_output: Optional[str] = None
    finished_output: Optional[str] = None
    
class GVProcess():

    def __init__(self, json_str:str, pdf_upload:str, pdf_watchlist):
        
        # Step 2: Create the Graph
        # Next, we create a new instance of StateGraph with our GraphState structure.
        self._json_str = json_str
        self._pdf_upload = pdf_upload
        self._pdf_watchlist = pdf_watchlist
        
        self.mService = MicroService(json_str, pdf_upload, pdf_watchlist)
        self.workflow = StateGraph(GraphState)
        
        ###################################################################
        # State switch configuration
        
        self._start = True
        self._monitoring_performed = True # SIMPLE genAI4Dev Use Case
        self._market_conditions_analysed = False # COMPLEX genAI4Dev Use Case
        self._transactions_analysed = False # COMPLEX genAI4Dev Use Case
        self._switches_compared = False # COMPLEX genAI4Dev Use Case
        self._lstm_executed = False
        
        ##################################################################
        
        # Step 4: Add Nodes to the Graph
        # We add our nodes to the graph and define the flow using edges and conditional edges.
        
        self.workflow.add_node("start", self.perform_monitoring)
        self.workflow.add_node("monitoring_performed", self.analyze_market_conditions)
        self.workflow.add_node("market_conditions_analyzed", self.analyze_transactions)
        self.workflow.add_node("transactions_analyzed", self.compare_switch)
        self.workflow.add_node("switches_compared", self.run_lstm)
        self.workflow.add_node("lstm_executed", self.finished)
        
        
        self.workflow.add_conditional_edges(
            "start",
            self.start_next_node,
            {
                "monitoring_performed": "monitoring_performed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "monitoring_performed",
            self.monitoring_performed_next_node,
            {
                "market_conditions_analyzed": "market_conditions_analyzed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "market_conditions_analyzed",
            self.market_conditions_analyzed_next_node,
            {
                "transactions_analyzed": "transactions_analyzed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "transactions_analyzed",
            self.transactions_analyzed_next_node,
            {
                "switches_compared": "switches_compared",
            }
        )
        
        self.workflow.add_conditional_edges(
            "switches_compared",
            self.switches_compared_next_node,
            {
                "lstm_executed": "lstm_executed",
            }
        )
        
        
        """
        self.workflow.add_conditional_edges(
            "start",
            self.start_next_node,
            {
                "market_conditions_analyzed": "market_conditions_analyzed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "market_conditions_analyzed",
            self.market_conditions_analyzed_next_node,
            {
                "transactions_analyzed": "transactions_analyzed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "transactions_analyzed",
            self.transactions_analyzed_node,
            {
                "monitoring_performed": "monitoring_performed",
            }
        )
        
        self.workflow.add_conditional_edges(
            "monitoring_performed",
            self.monitoring_performed_next_node,
            {
                "switches_compared": "switches_compared",
            }
        )
        
        self.workflow.add_conditional_edges(
            "switches_compared",
            self.switches_compared_next_node,
            {
                "lstm_executed": "lstm_executed",
            }
        )
        
        """
        
        # Step 5: Set Entry and End Points
        # We set the entry point for our graph and define the end points.
        
        self.workflow.set_entry_point("start")
        self.workflow.add_edge('start', 'monitoring_performed')
        self.workflow.add_edge('monitoring_performed', 'market_conditions_analyzed')
        self.workflow.add_edge('market_conditions_analyzed', 'transactions_analyzed')
        self.workflow.add_edge('transactions_analyzed', 'switches_compared')
        self.workflow.add_edge('switches_compared', 'lstm_executed')
        self.workflow.add_edge('lstm_executed', END)

    # Step 3: Define Nodes
    # We define nodes for classifying the input, handling greetings, and handling search queries.

    def perform_monitoring(self, state):
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

    def handle_perform_monitoring(self):
      # print(f"    ....perform_monitoring by calling crewAI....")
      
      if self._monitoring_performed:
        result = self.mService.perform_monitoring(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def analyze_market_conditions(self, state):
      print(f"\n    GVProcess - 2. analyze_market_conditions...")
      
      _crewAI_response = self.handle_analyze_market_conditions()
      # print(f"analyze_market_conditions response: {_crewAI_response}")
      if _crewAI_response == "Not activated":
        _analyze_market_conditions = "Not activated"
        print(f"\n        Not Activated")
      else:
        _analyze_market_conditions = "I've analyzed market conditions"
        print(f"\n        Successfully Processed")
      
      return {"analyze_market_conditions_output": _analyze_market_conditions}

    def handle_analyze_market_conditions(self):
      #print(f"    ....handle_analyze_market_conditions by calling crewAI....")
      
      if self._market_conditions_analysed :
        result = self.mService.analyze_market_conditions(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def analyze_transactions(self, state):
      print(f"\n    GVProcess - 3. analyze_transactions...")
      
      _crewAI_response = self.handle_analyze_transactions()
      # print(f"analyze_transactions response: {_crewAI_response}")
      if _crewAI_response == "Not activated":
        _analyze_transactions = "Not activated"
        print(f"\n        Not Activated")
      else:
        _analyze_transactions = "I've analyzed transactions"
        print(f"\n        Successfully Processed")

      return {"analyze_transactions_output": _analyze_transactions}

    def handle_analyze_transactions(self):
      # print(f"    ....handle_analyze_transactions by calling crewAI....")

      if self._transactions_analysed:
        result = self.mService.analyze_transactions(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def compare_switch(self, state):
      print(f"\n    GVProcess - 4. compare_switch...")
      
      _crewAI_response = self.handle_compare_switch()
      #print(f"compare_switch response: {_crewAI_response}")
      
      if _crewAI_response == "Not activated":
        _compare_switch = "Not activated"
        print(f"\n        Not Activated")
      else:
        _compare_switch = "I've performed Switch comparison"
        print(f"\n        Successfully Processed")
      
      return {"compare_switch_output": _compare_switch}

    def handle_compare_switch(self):
      # print(f"    ....compare_switch by calling crewAI....")
      
      if self._switches_compared:
        result = self.mService.compare_switches(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def run_lstm(self, state):
      print(f"\n    GVProcess - 5. run_lstm...")
      
      _crewAI_response = self.handle_run_lstm()
      # print(f"run_lstm response: {_crewAI_response}")
      
      if _crewAI_response == "Not activated":
        _run_lstm = "Not activated"
        print(f"\n        Not Activated")
      else:
        _run_lstm = "I've run LSTM"
        print(f"\n        Successfully Processed")
        
      return {"run_lstm_output": _run_lstm}

    def handle_run_lstm(self):
      # print(f"....handle_run_lstm by calling crewAI....")
      
      if self._lstm_executed:
        result = self.mService.run_lstm(status_state="Waiting...")
      else:
        result = "Not activated"
      
      return result

    def finished(self, state):
      print("\n    5. Finishing...")
      print(f"        >perform_monitoring_output: {state.get('perform_monitoring_output', '').strip()}")
      print(f"        >analyze_market_conditions_output: {state.get('analyze_market_conditions_output', '').strip()}")
      print(f"        >analyze_transactions_output: {state.get('analyze_transactions_output', '').strip()}")
      print(f"        >compare_switch_output: {state.get('compare_switch_output', '').strip()}")
      print(f"        >run_lstm_output: {state.get('run_lstm_output', '').strip()}")
      
      _finished = "I've finished"
      return {"finished_output": _finished}

    def start_next_node(self, state):
      # print("        ...satisfies start_next_node rule")
      if state.get('perform_monitoring_output') != None :
        return "monitoring_performed" 

    def monitoring_performed_next_node(self, state):
      # print("        ...satisfies monitoring_performed_next_node rule")
      if state.get('analyze_market_conditions_output') != None: 
        return "market_conditions_analyzed" 

    def market_conditions_analyzed_next_node(self, state):
      # print("        ...satisfies market_conditions_analyzed_next_node rule")
      if state.get('analyze_transactions_output') != None: 
        return "transactions_analyzed" 

    def transactions_analyzed_next_node(self, state):
      # print("        ...satisfies transactions_analyzed_next_node rule")
      if state.get('compare_switch_output') != None:
        return "switches_compared" 

    def switches_compared_next_node(self, state):
      # print("        ...satisfies switches_compared_next_node rule")
      if state.get('run_lstm_output') != None:
        return "lstm_executed" 
      else:
        return "start"

    def execute(self):
      # Step 6: Compile and Run the Graph
      # Finally, we compile our graph and run it with some input.

      print(f"\n\nGVProcess...executing")
      app = self.workflow.compile()
      inputs = {
        "transactions": {self._json_str},
        "pdf_upload" : {self._pdf_upload}
        }
      result = app.invoke(inputs)
      
      print(f"\n Workflow execution complete.  result={result}")
      
      return result
