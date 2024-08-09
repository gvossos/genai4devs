import pandas as pd
import phoenix as px
from phoenix.trace import SpanEvaluations
from phoenix.trace.dsl import SpanQuery
from phoenix.evals import llm_classify, llm_generate, OpenAIModel
from datetime import datetime, timedelta
import json

def evaluate_traces():
    """
    Evaluates traces by querying spans from Phoenix, performing supported evidence and overall summary evaluations,
    and logging the results.
    """
    px_client = px.Client(endpoint="http://localhost:6006")
    trace_df = query_spans_from_phoenix(px_client)
    supported_evidence_eval_df = supported_evidence_eval(trace_df)
    overall_summary_eval_df = overall_summary_eval(trace_df)
    
    px.Client().log_evaluations(SpanEvaluations(eval_name="EvidenceSupported", dataframe=supported_evidence_eval_df))
    px.Client().log_evaluations(SpanEvaluations(eval_name="Summarization", dataframe=overall_summary_eval_df))

def query_spans_from_phoenix(px_client):
    """
    Queries spans from Phoenix within the last day, filtering for LLM spans containing specific content.

    Args:
        px_client: Phoenix client instance

    Returns:
        DataFrame containing filtered span data
    """
    start = datetime.now() - timedelta(days=1)
    end = datetime.now()
    query = SpanQuery().where("span_kind == 'LLM'").select(
        span_id="context.span_id",
        input="input.value",
        output="output.value",
    )
    trace_df = px_client.query_spans(query, start_time=start, end_time=end)
    return trace_df[
        trace_df['input'].str.contains("you are performance analyst", case=False, na=False) &
        ~trace_df['input'].str.contains('"lc": 1', case=False, na=False)
    ]

def extract_data_from_llm_spans(dataset, extraction_type):
    """
    Extracts specific data from LLM spans based on the extraction type.

    Args:
        dataset: DataFrame containing LLM span data
        extraction_type: Type of extraction to perform ('fact_sets_and_summaries' or 'overall_summary')

    Returns:
        DataFrame with extracted data
    """
    extracted_data = []
    for index, row in dataset.iterrows():
        try:
            output = json.loads(row['output'])
            if isinstance(output, dict) and 'choices' in output:
                content = output['choices'][0]['message']['content']
                if extraction_type == 'fact_sets_and_summaries':
                    extracted = extract_fact_sets_and_summaries(content, index)
                elif extraction_type == 'overall_summary':
                    extracted = extract_overall_summary(content, index)
                if extracted:
                    extracted_data.append(extracted)
            else:
                print(f"Row {index}: Output does not contain expected structure.")
        except Exception as e:
            print(f"Error processing row {index}: {str(e)}")
    return pd.DataFrame(extracted_data)

def extract_fact_sets_and_summaries(content, index):
    """
    Extracts fact sets and summaries from the content of an LLM span.

    Args:
        content: String content of the LLM span
        index: Index of the current row being processed

    Returns:
        Dictionary containing extracted fact sets and summaries, or None if extraction fails
    """
    split_text = content.split("####")
    if "Final Answer" in split_text[0]:
        fact_sets, summaries = [], []
        for entry in split_text[1:]:
            parts = entry.split("**Justification**")
            if len(parts) == 2:
                fact_sets.append(parts[0].strip())
                summaries.append(parts[1].strip())
        
        fact_set = ' '.join(fact_sets)
        summary = ' '.join(summaries)
        
        if fact_set != '' and summary != '':
            return {
                'context.span_id': index,
                'fact_set': fact_set,
                'summary': summary
            }
        return None
    return None

def extract_overall_summary(content, index):
    """
    Extracts the overall summary from the content of an LLM span.

    Args:
        content: String content of the LLM span
        index: Index of the current row being processed

    Returns:
        Dictionary containing extracted overall summary and full text, or None if extraction fails
    """
    overall_summary_index = content.find("### Overall Summary")
    if overall_summary_index != -1:
        overall_summary = content[overall_summary_index + len("### Overall Summary"):].strip()
        full_text = content[:overall_summary_index].strip()
        return {
            'context.span_id': index,
            'overall_summary': overall_summary,
            'full_text': full_text
        }
    print(f"Row {index}: '### Overall Summary' not found in content.")
    return None

def supported_evidence_eval(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates whether summaries are supported by their corresponding fact sets.

    Args:
        dataset: DataFrame containing LLM span data

    Returns:
        DataFrame with evaluation results merged with the original data
    """
    supported_evidence_prompt = """
    Given the following fact set and summary, determine if the summary is supported by the fact set.
    Fact set: {fact_set}
    Summary: {summary}
    
    Only answer with "supported" or "hallucinated". Supported means that the summary is supported by the fact set. 
    Hallucinated means that the summary is not supported by the fact set.
    """
    extracted_df = extract_data_from_llm_spans(dataset, 'fact_sets_and_summaries')
    supported_evidence_eval = llm_classify(
        dataframe=extracted_df,
        template=supported_evidence_prompt,
        model=OpenAIModel('gpt-4o'),
        concurrency=20,
        rails=["supported", "hallucinated"],
        provide_explanation=True,
    )
    return pd.merge(extracted_df, supported_evidence_eval, left_index=True, right_index=True)

def overall_summary_eval(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates the quality of overall summaries.

    Args:
        dataset: DataFrame containing LLM span data

    Returns:
        DataFrame with evaluation results merged with the original data
    """
    summary_quality_prompt = """
    Given the following full text and overall summary, determine the quality of the overall summary.
    Full text: {full_text}
    Overall summary: {overall_summary}
    
    Return a score between 0 and 100, where 100 is the best score. Only return a score, no other text.
    """
    extracted_df = extract_data_from_llm_spans(dataset, 'overall_summary')
    summary_quality_eval = llm_generate(
        dataframe=extracted_df,
        template=summary_quality_prompt,
        model=OpenAIModel('gpt-4o'),
        concurrency=20,
        output_parser=numeric_score_eval,
    )
    return pd.merge(extracted_df, summary_quality_eval, left_index=True, right_index=True)

def numeric_score_eval(output, row_index):
    """
    Parses the output of the summary quality evaluation into a numeric score.

    Args:
        output: String output from the LLM
        row_index: Index of the current row being processed

    Returns:
        Dictionary containing the parsed score, or None if parsing fails
    """
    try:
        score = float(output.strip())
        if 0 <= score <= 100:
            return {"score": score}
        print(f"Row {row_index}: Score {score} is out of range (0-100).")
    except ValueError:
        print(f"Row {row_index}: Could not convert '{output}' to a number.")
    return None

if __name__ == "__main__":
    evaluate_traces()