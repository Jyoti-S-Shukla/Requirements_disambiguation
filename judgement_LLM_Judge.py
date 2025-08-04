import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
# from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase
import os
# from openai import OpenAI
import pandas as pd
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
# from deepeval.metrics.g_eval import Rubric

file_path = r"file_path.xlsx"
file = pd.read_excel(file_path)
requirements = file["Sentences"].tolist()
ambiguity = file['Ambiguity Type'].tolist()
explanations = file["Justification"].tolist()

OpenAI_Key = YOUR_OPENAI_KEY
os.environ["OPENAI_API_KEY"] = OpenAI_Key

judgements= []
reasonings = []
success = []
for counter, (requirement, classification, explanation) in enumerate(zip(requirements, ambiguity, explanations)):
  test_case = LLMTestCase(
    input=f"{requirement}",
    actual_output=f'''Analysis={{
      "Ambiguous": "{classification}",
      "Explanation": "{explanation}",
  }}''',
    # expected_output ="Not ambiguous."
    #retrieval_context=["The direction for sun rising is west", 'The direction for sun rising is east'],
  )
  correctness_metric = GEval(
      name="Correctness",
      criteria="Determine whether the ambiguity type classification in the actual output is correct or incorrect.",
      evaluation_steps=[
          "Read the provided input requirement sentence carefully",
          "Read the actual output comprising of ambiguity type(s) and explanation for the respective ambiguity classification. Based on the input, judge whether the ambiguity classification(s) is correct or not.",
          "For better understanding, go through the explanation provided for the ambiguity classification and make a final judgement whether the ambiguity classification is correct or not.",
          "Be unbiased and strictly go through the input and output before ensuring your judgement."
      ],
      model = 'gpt-4o-mini',
      # async_mode = False,
      evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
      # rubric=[
      #     Rubric(score_range=(0,0), expected_outcome="Incorrect Judgement."),
      #     Rubric(score_range=(1,1), expected_outcome="Correct Judgement."),
      # ],
      threshold= 0.75)
  correctness_metric.measure(test_case)
  judgements.append(correctness_metric.score)
  reasonings.append(correctness_metric.reason)
  success.append(correctness_metric.is_successful())
  if counter%10==0:
    print("Input number:", counter, "Input test case:\n", test_case)
    print(correctness_metric.score, correctness_metric.reason, correctness_metric.is_successful())

file["Judgement"] = judgements
file['Judge_reasoning'] = reasonings
file['Passed'] = success
file.to_excel(r"file_path_judged.xlsx")
