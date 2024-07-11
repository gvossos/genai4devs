# CrewAI SaaS for Stock Analysis using LLMs
## Introduction
This project extends the crewAI Stock Analysis project to create a SaaS (Software as a Service) project using the CrewAI framework to automate the process of analyzing a stock. CrewAI orchestrates autonomous AI agents, enabling them to collaborate and execute complex tasks efficiently using a SaaS architecture.  SaaS is enabled via fastAPI micro-services.

CrewAI SaaS inspiration and reference can be found here - `https://www.youtube.com/watch?v=pFZHpFuzcBE&t=854s` by @Kno2gether https://www.youtube.com/@Kno2gether 


By [@gvossos](https://github.com/gvossos/)

- [VM Setup Crew AI](#vm-setup-crew-ai)
- [CrewAI Framework](#crewai-framework)
- [Running the script](#running-the-script)
- [Details & Explanation](#details--explanation)
- [Using GPT 3.5](#using-gpt-35)
- [Using Local Models with Ollama](#using-local-models-with-ollama)
- [Contributing](#contributing)
- [Support and Contact](#support-and-contact)
- [License](#license)

## VM Setup Crew AI
Create a virtual environment for CrewAI.

- **conda**: `conda create --name tradingview python=3.10.10 pip`
- **conda**: `activate tradingview
- **conda**: cd tradingview


## CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example, these agents work together to give a complete stock analysis and investment recommendation

## Running the Script
Note that this SaaS uses a local open-source LLM so you you will need to refer to the VM Setup Local LLM section.


- **Configure Environment**: Copy ``.env.example` and set up the environment variables for [Browseless](https://www.browserless.io/), [Serper](https://serper.dev/), [SEC-API](https://sec-api.io) and [OpenAI](https://platform.openai.com/api-keys)
- **Install crewAI Dependencies**: Run `poetry install --no-root`.
- **Install SaaS Dependencies**: `conda install fastapi uvicorn`
- `pip install python-dotenv`
- `pip install --upgrade pydantic==2.4.2`
- `pip install jija2`
- `pip install langchain_openai`
- `pip install -U langchain-community faiss-cpu langchain-openai tiktoken`
- `pip install gradio`
- **Execute the crewAI SaaS Service**: Run `python main.py` and input the name of your stock.

## VM Setup local LLM
Create a virtual environment for local LLM.  Note - you will need at least 30GB RAM!

- **conda**: `conda create --name textgen python=3.10.10 pip`
- **conda**: `activate textgen
- **conda**: cd text-generation-webui


## CrewAI Framework
CrewAI is designed to facilitate the collaboration of role-playing AI agents. In this example, these agents work together to give a complete stock analysis and investment recommendation

## Running the Script
Note that this SaaS uses a local open-source LLM so you you will need to refer to the VM Setup Local LLM section.


- **Configure Environment**: Copy ``.env.example` and set up the environment variables for [Browseless](https://www.browserless.io/), [Serper](https://serper.dev/), [SEC-API](https://sec-api.io) and [OpenAI](https://platform.openai.com/api-keys)
- **Install crewAI Dependencies**: Run `poetry install --no-root`.
- **Install SaaS Dependencies**: `conda install fastapi uvicorn`
- `pip install python-dotenv`
- `pip install --upgrade pydantic==2.4.2`
- `pip install jija2`
- `pip install langchain_openai`
- `pip install -U langchain-community faiss-cpu langchain-openai tiktoken`
- `pip install gradio`
- **Execute the crewAI SaaS Service**: Run `python main.py` and input the name of your stock.
## Details & Explanation
- **Running the Script**: Execute `python main.py`` and input your idea when prompted. The script will leverage the CrewAI SaaS framework to process the idea and generate a landing page.
Access the web-page by clicking on https://127.0.0.1:8000 link from the console.
- **Key Components**:
  - `./main.py`: Main script file.
  - `./stock_analysis_tasks.py`: Main file with the tasks prompts.
  - `./stock_analysis_agents.py`: Main file with the agents creation.
  - `./tools`: Contains tool classes used by the agents.

## Using LLMs
CrewAI allow you to pass an llm argument to the agent construtor, that will be it's brain, so changing the agent to use another LLM like GPT-3.5 instead of GPT-4 is as simple as passing that argument on the agent you want to use that LLM (in `main.py`).
```python
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model='gpt-3.5') # Loading GPT-3.5

def local_expert(self):
	return Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      llm=llm, # <----- passing our llm reference here
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ]
    )
```

## Using Local Models with Ollama
The CrewAI framework supports integration with local models, such as Ollama, for enhanced flexibility and customization. This allows you to utilize your own models, which can be particularly useful for specialized tasks or data privacy concerns.

### Setting Up Ollama
- **Install Ollama**: Ensure that Ollama is properly installed in your environment. Follow the installation guide provided by Ollama for detailed instructions.
- **Configure Ollama**: Set up Ollama to work with your local model. You will probably need to [tweak the model using a Modelfile](https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md), I'd recommend adding `Observation` as a stop word and playing with `top_p` and `temperature`.

### Integrating Ollama with CrewAI
- Instantiate Ollama Model: Create an instance of the Ollama model. You can specify the model and the base URL during instantiation. For example:

```python
from langchain.llms import Ollama
ollama_openhermes = Ollama(model="agent")
# Pass Ollama Model to Agents: When creating your agents within the CrewAI framework, you can pass the Ollama model as an argument to the Agent constructor. For instance:

def local_expert(self):
	return Agent(
      role='The Best Financial Analyst',
      goal="""Impress all customers with your financial data 
      and market trends analysis""",
      backstory="""The most seasoned financial analyst with 
      lots of expertise in stock market analysis and investment
      strategies that is working for a super important customer.""",
      verbose=True,
      llm=ollama_openhermes, # Ollama model passed here
      tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
      ]
    )
```

### Advantages of Using Local Models
- **Privacy**: Local models allow processing of data within your own infrastructure, ensuring data privacy.
- **Customization**: You can customize the model to better suit the specific needs of your tasks.
- **Performance**: Depending on your setup, local models can offer performance benefits, especially in terms of latency.

## License
This project is released under the MIT License.
