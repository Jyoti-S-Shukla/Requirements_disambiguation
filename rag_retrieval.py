from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
import os, torch, re, json
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import pandas as pd
import os, re
from langchain.llms import HuggingFacePipeline
from huggingface_hub import login
login(token="hf_**")

def create_vectorstore(document, similarity_top_k=3, score_threshold=0.6):
    text = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300).split_documents(document) # Split documents into chunks
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", encode_kwargs={"normalize_embeddings": True}) # Load embedding model
    vectorstore = Chroma.from_documents(text, embeddings) # Create a vectorstore    
    retriever = vectorstore.as_retriever(similarity_top_k=similarity_top_k, score_threshold=score_threshold) # Create retriever
    return vectorstore, retriever

def initialize_llm_and_chain(model_name = "mistralai/Mistral-7B-Instruct-v0.3", max_new_tokens=150, repetition_penalty=1.5, retriever=None):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Create the text generation pipeline
    pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty, torch_dtype=torch.float16, return_full_text='False', do_sample=True, temperature=0.01)# device_type="auto")  # Uncomment if needed)
    # Create the local LLM using the HuggingFace pipeline
    local_llm = HuggingFacePipeline(pipeline=pipe)
    # Define the prompt template
    template = """
    You are an assistant for question-answering tasks solving ambiguity-related issues in the input requirement statement with the help of following relevant contexts. Study the ambiguous element present in the question and generate answer only from the given contexts. If you don't find the answer in the context, just respond 'NA' strictly and no in ambiguity resolved segment. Your answer should comprise of two parts: a) Answer for the query (NA if not found in context) b) ambiguity resolved (yes/no) be definitive in your analysis and restrict answer to maximum 30 words. Use the provided context only to answer the following question. DO NOT HALLUCINATE.

    <context>
    {context}
    </context>

    Question: {input}
    Answer: 
    
    """
    # {{\\"Answer\\": \\"\\" ,
    # \\"Ambiguity resolved\\": \\"\\" , 
    # \\"Justification\\": \\"\\"}}'
    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(template)
    # Create a document chain
    doc_chain = create_stuff_documents_chain(local_llm, prompt)
    # Create a retrieval chain if a retriever is provided
    if retriever is not None:
        chain = create_retrieval_chain(retriever, doc_chain)
        return local_llm, prompt, chain

    return local_llm, prompt

def extract_questions(row):
    questions = re.split(r",\s*['\"]?Q", row)

    # Clean up the questions by stripping whitespace and handling any additional text
    cleaned_questions = [q.strip().replace("'", "").replace('"', '') for q in questions if q]

    # Ensure we have at least two questions
    if len(cleaned_questions) >= 2:
        Q1 = cleaned_questions[0]
        Q2 = "Q"+cleaned_questions[1]
        print(f"Q1: {Q1}")
        print(f"Q2: {Q2}")
        return Q1, Q2
    else:
        print("Not enough questions found.")
        return None, None

# Function to process and invoke chain for ambiguous rows
def process_ambiguous_requirements(df, chain, retriever, prompt, indices):
    # Create new columns in DataFrame for the answers and documents if they do not already exist
    columns_to_create = ['Q1', 'Q2', 'doc_Q1', 'doc_Q2']
    for col in columns_to_create:
        if col not in df.columns:
            df[col] = ""
    
    # Iterate through the specified indices
    for idx in indices:
        row = df.iloc[idx]  # Get the row from the original DataFrame
        ambiguity = row['Ambiguity']
        if ambiguity == "Ambiguous":
            Q1, Q2 = extract_questions(row['PQs'])
            requirement = row["Sentences"]

            # Initialize lists for answers and documents
            res, docs = [], []

            # Sequentially invoke the chain for Q1 and Q2
            for question in [Q1, Q2]:
                if question:  # Ensure the question is not empty
                    query = f"{requirement}, in this statement, {question}"
                    response = chain.invoke({"input": query})

                    # Get the answer from the response
                    ans = response.get('answer', '')
                    doc = retriever.get_relevant_documents(query)

                    if ans:
                        if "Assistant:" in ans:
                            print(f"Answer to {question}: {ans.split('Assistant:')[1]}")
                            res.append(ans.split('Assistant:')[1])
                        else:
                            print(f"Answer to {question}: {ans.split('Answer:')[1]}")                        
                            res.append(ans.split('Answer:')[1] if ans else "")
                    else:
                        res.append("")
                    docs.append(doc)
                else:
                    res.append("")
                    docs.append("")

            # Update the original DataFrame
            df.at[idx, 'Q1'] = f"{Q1}" if res else ""
            df.at[idx, 'Q2'] = f"{Q2}" if res else ""
            df.at[idx, 'Answer_1'] = f"{res[0]}" if res else ""
            df.at[idx, 'Answer_2'] = f"{res[1]}" if res else ""
            df.at[idx, 'doc_Q1'] = docs[0] if len(docs) > 0 else ""
            df.at[idx, 'doc_Q2'] = docs[1] if len(docs) > 1 else ""


try:
    # Folder where PDFs are stored
    # pdf_folder = files_path
    pdf_folder ="/workspace/path/PDFs/"
    pdfs= os.listdir(pdf_folder)
    print(pdfs,"\n", '-'*40)
    req_sheet_path = "/workspace/file_path.xlsx"
    # Read the main DataFrame
    df = pd.read_excel(req_sheet_path)

    # Group by document_name
    grouped_rows = {doc: data for doc, data in df.groupby('Document_Name')}

    # Integrate data and RAG chain
    for doc_name, group_df in grouped_rows.items():
        pdf_folder = "/workspace/data/Jyoti/Ambiguity/PURE/PURE_420_Ambiguity_Project/PDFs/"
        pdf_path = os.path.join(pdf_folder, f"{doc_name}.pdf")

        if os.path.exists(pdf_path):
            document = PyPDFLoader(pdf_path).load()
            print(f"Loaded PDF for document: {doc_name} with {len(group_df['Sentences'])} requirements")
            vectorstore, retriever = create_vectorstore(document)
            local_llm, prompt, chain = initialize_llm_and_chain(retriever=retriever)

            # Get the indices of the rows in the original DataFrame
            indices = group_df.index.tolist()
            process_ambiguous_requirements(df, chain, retriever, prompt, indices)
            # Release GPU memory
            del document, vectorstore, retriever, local_llm, prompt, chain
            torch.cuda.empty_cache()
        else:
            print(f"PDF file not found for document: {doc_name}")

    # Save the updated DataFrame to a new Excel file
    output_file = '/workspace/file_path__retrieved_answers_mistral.xlsx'
    df.to_excel(output_file, index=False)
    print(f"Updated Excel file saved at: {output_file}")

except Exception as e:
    print(f'Error:', str(e))
