print("Starting......")
from langchain import PromptTemplate

import pandas as pd
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import streamlit as st
import numpy as np




## Function to make Vector DB (Using Faiss and opensource Huggingface Embeddings )

def make_faiss_index(path):
    loader = DirectoryLoader(path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    ## considering first 2 chapter, manually found out the pages
    start = 18
    end = 70
    documents = documents[start:end]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    Faiss_Index_Path = 'vecotor_db_index/db_faiss'

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(Faiss_Index_Path)







Faiss_Index_Path = 'vecotor_db_index/db_faiss'
#DB_FAISS_PATH = 'vectorstore/db_faiss'

base_prompt_template = """You are AI assistant which specilizes in providing accurate answers based on the question. Use the following pieces of information to answer the user's question.
Please try to give exact answers. Please say I dont know if u dont know the answer. Please dont give any random answers.

Context: {context}
Question: {question}

Only return the helpful answer below.
Helpful answer:
"""

def base_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=base_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

#Loading LLM model will beed to change model based on performance
def load_llm_model():
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#make_faiss_index("")

## Run Streamlit application for QnA and change prompt based on that

if __name__ == "__main__":
    print("cool")

    prompt = st.chat_input("Please ask your question")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(Faiss_Index_Path, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm_model()
    qa_prompt = base_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    if prompt:
        answer = qa(prompt)["result"]
        st.write(f"Question: {prompt}")
        st.write(f"Answer from LLM models : {answer}")
    






## Function to call from Notebooks
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(Faiss_Index_Path, embeddings,allow_dangerous_deserialization=True)
    llm = load_llm_model()
    qa_prompt = base_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa




#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


#print(final_result("What is Genome")["result"])


