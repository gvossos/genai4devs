import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.evals import llm_generate, OpenAIModel
import json

PHOENIX_ENDPOINT = "http://localhost:6006"

def retrieve_spans_from_phoenix():
    px_client = px.Client(endpoint=PHOENIX_ENDPOINT)

    query = SpanQuery().where(
        "name == 'CrewAgentExecutor'",
    ).select(
        span_id="context.span_id",
        input="input.value",
        output="output.value",
    )

    spans_df = px_client.query_spans(query)
    spans_df['input'] = spans_df['input'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    spans_df = spans_df.reset_index()
    new_df = pd.DataFrame({
        'context.span_id': spans_df['context.span_id'],
        'output': spans_df['output']
    })
    flattened_input = pd.json_normalize(spans_df['input'])
    new_df = pd.concat([new_df, flattened_input], axis=1)
    new_df.set_index('context.span_id', inplace=True)
    
    spans_df = new_df
    return spans_df
    
def evaluate_spans(spans_df):
    def split_input(input_str):
        parts = input_str.split("Agent Tool parameters are", 1)
        return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (input_str.strip(), "")

    def output_parser(response, response_index):
        try:
            parsed_response = json.loads(response)
            score = int(parsed_response.get('score', 0))
            explanation = parsed_response.get('explanation', '')
            return {'score': score, 'explanation': explanation}
        except json.JSONDecodeError:
            print(f"Row {response_index}: Failed to parse response as JSON: {response}")
        except Exception as e:
            print(f"Row {response_index}: An error occurred while parsing the response: {str(e)}")
        return None

    spans_df['task_description'], spans_df['tool_parameters'] = zip(*spans_df['input'].apply(split_input))
    
    llm_as_a_judge_prompt = """
    You are a fair and impartial judge evaluating how well agents complete their assigned tasks.
    Use the task description and the agent's output below for your evaluation.
    Return a numeric score and explanation as a valid JSON response.
    
    Scoring:
    0: Agent did not return anything.
    1: Agent gave a response but did not meet task expectations.
    2: Agent completed the task adequately.
    3: Agent completed the task perfectly with no room for improvement.
    
    ----
    BEGIN DATA
    [task description]: {task_description}
    [output]: {output}
    END DATA
    ----
    
    Example response:
    'score': 1, 'explanation': 'the agent gave a response but did not address all of the criteria listed in the task description'
    """
    
    evaluation_results = llm_generate(
        dataframe=spans_df,
        template=llm_as_a_judge_prompt,
        model=OpenAIModel('gpt-4o'),
        concurrency=20,
        output_parser=output_parser,
    )
    
    return pd.merge(spans_df, evaluation_results, left_index=True, right_index=True)

def log_evaluation_results_to_phoenix(spans_df):
    px.Client(endpoint=PHOENIX_ENDPOINT).log_evaluations(SpanEvaluations(eval_name="Agent Response Quality", dataframe=spans_df))

if __name__ == "__main__":
    spans_df = retrieve_spans_from_phoenix()
    spans_df = evaluate_spans(spans_df=spans_df)
    log_evaluation_results_to_phoenix(spans_df=spans_df)