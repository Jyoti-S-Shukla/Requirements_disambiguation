import torch
from transformers import AutoTokenizer, pipeline
import pandas as pd
import re, json
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
login(token="hf_**")
# device = "cuda" # the device to load the model onto

model_name = "Qwen/Qwen2-7B-Instruct"
# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name ="meta-llama/Llama-3.1-8B-Instruct"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# model_name = "deepseek-ai/deepseek-llm-7b-chat"
# model_name = "microsoft/phi-4"
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast =False)
file_path = "/workspace/file_path"
pure_data = pd.read_excel(file_path)
requirements = pure_data["Sentences"].tolist()
count =0
responses =[]
for req in requirements:
    prompt = ("""
          You are a requirement analyst and your job is to check whether the given requirement sentence is ambiguous in any possible way which can lead to wrong or multiple interpretations. If the requirement sentence is mostly clear in its context and meaning, render it unambiguous. If it contains some major ambiguity only then mark it "True" for ambiguity and also give an explanation of why you think it is ambiguous or unambiguous. Second, if it is Ambiguous, based on the explanation given by you, develop a list of at most 2 probing questions to be asked by the business analyst which can identify the source of ambiguity in the statement.
Here are some examples for better understanding:

 Requirement Statement: The website shall protect itself from intentional abuse  and notify the administrator at all occurrences.'
 Analysis={
    "Ambiguous": "True",
    "Explanation": "The term 'all occurrences' is ambiguous as it is not clear whether it means every single occurrence or a certain threshold of occurrences.",
    "Probing Questions required":"Yes",
    "PQs":[
        "Q1.: The term 'all occurrences' means every single occurrence or a certain threshold of occurrences?",
        "Q2.: Is there any set guideline to determine 'intentional' abuse?"
    ]
    }
Requirement Statement: The system shall have a MDI form that allows for the viewing of the graph and the data table.
Analysis={
    "Ambiguous": "False",
    "Explanation": "The requirement is clear about the system having a MDI form with two components: graph and data table. The term MDI form is well defined in the context of windows application development.",
    "Probing Questions required":"No",
    "PQs":[ " "]
}

Requirement Statement: The product shall have a conservative and professional appearance.'
Analysis={
    "Ambiguous": "True",
    "Explanation": "The term 'conservative' and 'professional appearance' lacks definition and is subjective leading to ambiguity."
    "Probing Questions required":"Yes",
    "PQs":[
        "Q1.: The term 'conservative' is subjective and hence ambiguous. What defines the product to be conservative?",
        "Q2.: The term 'professional appearance' is also subjective and open to multiple interpretations. What are the criterias to render the appearance as professional?""
    ]
}
Provide the JSON response for the following requirement statement in the format given below and Remember you need to ask the Probing Question(s) only if the requirement is Ambiguous. Restrict your answer to the given below format strictly:
Analysis={
    "Ambiguous": "True" or "False",
    "Explanation":"",
    "Probing Questions required": Yes if Ambiguous or No if not Ambiguous,
    "PQs":["Q1.",
    "Q2.",
    ..
    "QN."]
}
          """
                  f"Requirement sentence:'{{{req}}}`")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=400
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("Requirement", count, ":", req, response)
    count+=1
    responses.append(response)

def extract_answer_and_explanation(text_list):
    ambiguous_answers = []
    pq_list = []  # List to store PQs
    missing_ambiguous_indices = []
    explanations =[]

    # Process each text from the list
    for idx, text in enumerate(text_list):
        # Check for the presence of "Ambiguous" and extract the answer ("True" or "False")
        ambiguous_match = re.search(r'"Ambiguous":\s*"([^"]+)"', text)
        if ambiguous_match:
            ambiguous_answer = ambiguous_match.group(1)
            ambiguous_answers.append(ambiguous_answer)
        else:
            # If "Ambiguous" is not found, store the index
            missing_ambiguous_indices.append(idx)
            ambiguous_answers.append("MISSING")

        # Check for "PQs" (or "PQq") and extract the list of PQs
        pq_match = re.search(r'"PQs":\s*\[(.*?)\]', text, re.DOTALL)  # Match the PQ array content
        if pq_match:
            pq_text = pq_match.group(1)
            # Extract individual PQs from the matched group
            pq_list.append([pq.strip().strip('"') for pq in pq_text.split(',')])
        else:
            pq_list.append([])  # If no PQs are found, append an empty list
        # Check for "Explanation" and extract the explanation text
        explanation_match = re.search(r'"Explanation":\s*"([^"]+)"', text)
        if explanation_match:
            explanation_text = explanation_match.group(1)
            explanations.append(explanation_text)
        else:
            explanations.append("MISSING")

    return ambiguous_answers, pq_list, explanations, missing_ambiguous_indices

# Extract data
ambiguous_answers, pq_list, explanations, missing_ambiguous_indices = extract_answer_and_explanation(responses)

# Print the results
# print("Ambiguous Answers:", ambiguous_answers)
# print("PQs List:", pq_list)
print("Missing Ambiguous Indices:", missing_ambiguous_indices)

# pure_data["Requirement"]= requirements
pure_data["Response"] = responses
pure_data['Explanation']= explanations
pure_data["Ambiguous"]=ambiguous_answers
pure_data["PQs"]= pq_list
pure_data.to_excel("/workspace/file_path_2.xlsx", index=False)
