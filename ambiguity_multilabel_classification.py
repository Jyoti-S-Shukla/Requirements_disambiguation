import torch
from transformers import AutoTokenizer, pipeline
import pandas as pd
import re, json

#.......Login.........................
from huggingface_hub import login
login(token="hf_**")

#...........load model and build pipeline..............
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

pipe = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

#......... generate output.................

def get_llama_response(prompt: str) -> str:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    sequences = pipe(
        prompt,
        do_sample=True,
        top_p=0.2,
        num_return_sequences=1,
        eos_token_id=terminators,
        pad_token_id = tokenizer.pad_token_id,
        max_new_tokens=80,
        temperature =0.2,
    )
    response = sequences[0]['generated_text']
    return response

    import pandas as pd
pure = pd.read_excel("File_path")
requirements = pure["Requirement Text"].tolist()
count = 0
responses =[]
for req in requirements:
  req = req
  prompt = ("""
          You are a requirement analyst and your job is to Classify the requirement sentences into the ambiguity classes: lexical, structural, semantic, vagueness and pragmatic. The ambiguity classes are explained below for your reference:

          Consider the following examples for better understanding the classification types of ambiguities:

          Requirement sentence 1: "The system should 'validate' user data before processing."
          Ambiguity Class :  Lexical ambiguity
          Explanation: here "Validate" could mean basic data type checks or more rigorous data integrity checks depending on the interpretation.

          Requirement sentence 2: ""The system should allow the user to enter their details and view the results.""
          Ambiguity Class: Structural ambiguity
          Explanation: Here, it's unclear if user enters their details first, and after that, they can view the results or user can enter their details and immediately view the results without a clear sequence of actions.

          Requirement sentence 3: “The system should allow users to access data remotely.”
          Ambiguity Class: Semantic ambiguity
          Explanation: Here, the term "Remotely" has multiple meanings,it  could mean accessing data over the internet, using a remote desktop tool, or from a distant server.

          Requirement sentence 4: "The system needs to be highly reliable."
          Ambiguity Class: Vagueness
          Explanation: Here the term "highly reliable" is imprecise (lacks detail) and don't specify response time thresholds or acceptable failure rates, leaving room for interpretation due to lack of quantifiable metrics.

          Requirement sentence 5: "The report should be visually appealing."
          Ambiguity Class: Pragmatic ambiguity
          Explanation: Here, intent or the context of a statement is unclear; the term "visually appealing" is subjective and can be interpreted differently by different designers depending on their experience and expectations.

          Carefully read the requirement sentence in the single backticks and classify it in the above-mentioned classes of ambiguity.  Give the output in a clear and crisp form and strictly in the following format:
        "Output": {
                    "Ambiguity_class": "",
                    "Justification": "",
                    "Possible Interpretations":"Rank 1: 
                                                Rank 2:
                                                Rank 3:"
                }

          """
                  f"Requirement sentence:'{{{req}}}`")
  response= get_llama_response(prompt)
  responses.append(response[len(prompt):])
  print("Requirement ",count, ":", req, response[len(prompt):])
  count+=1

pure['Responses'] = responses
pure.to_excel("/workspace/data/Jyoti/Dataset_notes/Ambiguity/PROMISE_classes_binary.xlsx", index=False)
#   print("*"*100)
