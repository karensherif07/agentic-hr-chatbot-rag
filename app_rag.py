import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS      
from langchain_groq import ChatGroq                    
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() 

api_key = os.getenv("GROQ_API_KEY")


custom_prompt_template = """Use the following pieces of context to answer the user's question. 
If the question is about eligibility for a specific employee group (like Faculty or exempt Admin), 
look carefully for explicit exclusions or negative constraints, such as sentences starting 
with "do not receive," "are not eligible," "is not obligated," or "program does not apply." 

Always prioritize specific eligibility criteria (FTE requirements, length of service) 
and check the FAQ sections provided in the context for clarifications.

If the context explicitly states that a group does NOT receive a benefit or is NOT eligible, 
your answer must reflect that.

Context: {context}
Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

@st.cache_resource
def setup_rag_system():
    loader = PyPDFLoader("policy.pdf")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )


    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True
    )

qa = setup_rag_system()

st.title("RAG HR Chatbot")
user_question = st.text_input("Ask a question about the policy:")

if user_question:
    with st.spinner("Searching the policy..."):
        response = qa.invoke({"query": user_question})
        st.write("### Answer:")
        st.write(response["result"])
        if "source_documents" in response and response["source_documents"]:
            with st.expander("View Source Citations"):
                # Get unique pages
                source_pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in response["source_documents"]])))
                st.write(f"Information retrieved from PDF pages: **{', '.join(map(str, source_pages))}**")
                
                for i, doc in enumerate(response["source_documents"]):
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Snippet {i+1} (Page {page_num}):**")
                    st.info(doc.page_content)
        else:
            st.warning("Sources were not found in the response. Ensure 'return_source_documents=True' is set in the chain.")
        