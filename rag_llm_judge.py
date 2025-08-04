from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate
from deepeval import evaluate
import os
from openai import OpenAI
import pandas as pd
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
# from deepeval.metrics.g_eval import Rubric
import pandas as pd

OpenAI_Key = YOUR_OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OpenAI_Key

class CustomTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""In the given text, the answer for query comprising the ambiguity in requirement sentence is provided. Study the text and analyze whether the provided answer is relevant with the query and if the ambiguity is solved in the input requirement. If the answer resolves the query, it is relevant else it is irrelevant.

        Text:
        {actual_output}

        JSON:
        """

def analyze_rag_output(input_file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(input_file)
    # Create new columns in DataFrame for the answers and documents if they do not already exist
    columns_to_create = ['Ar1', 'Ar1_reason', 'Ar2', 'Ar2_reason', 'Cr1', 'Cr1_reason', 'Cr2', 'Cr2_reason']
    for col in columns_to_create:
        if col not in df.columns:
            df[col] = ""

    # Iterate through the rows
    for idx, row in df.iterrows():
        # Check if the 'Ambiguous_qwen' column is TRUE
        if row["Ambiguity"] == "Ambiguous":
            print("Ambiguous requirement at", idx)
            Q1 = row["Q1"]
            Q2 = row["Q2"]
            Q1_Answer = row['Answer_1']
            Q2_Answer = row['Answer_2']
            rc1 = row['doc_Q1']
            rc2 = row['doc_Q2']
            
            # Initialize the result lists to store responses and docs
            ar = []
            cr = []
            ar_reason = []
            cr_reason = []

            # Sequentially invoke the chain for Q1, Q2, Q3
            for i, question in enumerate([Q1, Q2], 1):
                if pd.notna(question):  # Ensure the question is not NaN or empty
                    query = f"{row['Sentences']}, in this statement, {question}"
                    print(f"Sending query: {query}")
                    # Replace this with the actual output from your LLM application
                    actual_output = [Q1_Answer, Q2_Answer][i-1]  # Corrected index
                    retrieval_context = [rc1, rc2][i-1]  # Corrected index

                    # Answer Relevancy
                    print("*"*10 + "Answer Relevance" + "*"*10)
                    try:
                        test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieval_context])
                        ar_metric = AnswerRelevancyMetric(threshold=0.75, model='gpt-4o-mini', include_reason=True, evaluation_template=CustomTemplate)
                        ar_metric.measure(test_case)
                        ar_score = ar_metric.score
                        arm_reason = ar_metric.reason  # Answer relevancy metric reason
                        print("Through custom template, Answer Relevancy scores are:", ar_score)
                        print(ar_metric.is_successful())
                        ar.append(ar_score)
                        ar_reason.append(arm_reason)  # Reason for the answer relevancy score
                    except Exception as e:
                        error_message = f"Error in Answer Relevancy: {str(e)}"
                        print(error_message)
                        ar.append(-1)
                        ar_reason.append(error_message)  # Append error message

                    # Contextual Relevance
                    print("*"*10 + "Contextual relevance" + "*"*10)
                    try:
                        cr_metric = ContextualRelevancyMetric(threshold=0.75, model='gpt-4o-mini', include_reason=True)
                        test_case = LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieval_context])
                        cr_metric.measure(test_case)
                        cr_score = cr_metric.score
                        crm_reason = cr_metric.reason  # Contextual relevancy metric reason
                        print("Through custom template, Contextual Relevancy scores are:", cr_score)
                        print(cr_metric.is_successful())
                        cr.append(cr_score)
                        cr_reason.append(crm_reason)  # Reason for the contextual relevancy score
                    except Exception as e:
                        error_message = f"Error in Contextual Relevancy: {str(e)}"
                        print(error_message)
                        cr.append(-1)
                        cr_reason.append(error_message)  # Append error message

                else:
                    ar.append(-1)  # If the question is empty, append an empty answer
                    ar_reason.append("")  # If the question is empty, append an error message
                    cr.append(-1)  # If the question is empty, append an empty answer
                    cr_reason.append("")  # If the question is empty, append an error message

            # Assign answers to the DataFrame
            df.at[idx, 'Ar1'] = ar[0] if len(ar) > 0 else ""
            df.at[idx, 'Ar1_reason'] = ar_reason[0] if len(ar_reason) > 0 else ""
            df.at[idx, 'Ar2'] = ar[1] if len(ar) > 1 else ""
            df.at[idx, 'Ar2_reason'] = ar_reason[1] if len(ar_reason) > 1 else ""
            df.at[idx, 'Cr1'] = cr[0] if len(cr) > 0 else ""
            df.at[idx, 'Cr1_reason'] = cr_reason[0] if len(cr_reason) > 0 else ""
            df.at[idx, 'Cr2'] = cr[1] if len(cr) > 1 else ""
            df.at[idx, 'Cr2_reason'] = cr_reason[1] if len(cr_reason) > 1 else ""
        # Optionally, save the updated DataFrame to a new Excel file
    output_file = r"file_path_judge_outcomes.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Updated Excel file saved at: {output_file}")

input_file = r"file_path_retrieved_answers_mistral.xlsx"  # Path to your input Excel file
analyze_rag_output(input_file)
