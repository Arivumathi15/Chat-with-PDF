import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    Answer the questions as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    context:\n {context}?\n
    Question: \n{question}\n

    Answer:
"""
    model = ChatGoogleGenerativeAI(model ="gemini-pro",
                                   temperature = 0.3)
    
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def highlight_keywords(text, keywords):
    # Function to highlight keywords in text
    highlighted_text = text
    for keyword in keywords:
        highlighted_text = re.sub(r'\b({})\b'.format(re.escape(keyword)), r'<mark>\1</mark>', highlighted_text, flags=re.IGNORECASE)
    return highlighted_text

def search_and_highlight(pdf_docs, user_question):
    # Search PDF content for keywords and highlight them
    highlighted_docs = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        highlighted_pdf = []
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            highlighted_text = highlight_keywords(page_text, user_question.split())
            highlighted_pdf.append(highlighted_text)
        highlighted_docs.append(highlighted_pdf)
    return highlighted_docs


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs = True)
    
    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question from PDF file")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files and click on the submit & process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    # Search for keywords and highlight
                    highlighted_text = highlight_keywords(raw_text, user_question)
                    # Display highlighted text
                    st.text_area("Highlighted Text", value=highlighted_text, height=600)
                    # Process text chunks and store vectors
                    text_chunks = get_text_chunks(highlighted_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                else:
                    st.error("Please upload PDF files.")


if __name__ == "__main__":
    main()