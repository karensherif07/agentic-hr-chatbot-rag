import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader , PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS      
from langchain_groq import ChatGroq                    
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv() 
api_key = os.getenv("GROQ_API_KEY")


# Custom prompt template
# custom_prompt_template = """Use the following pieces of context to answer the user's question. 
# If the question is about eligibility for a specific employee group (like Faculty or exempt Admin), 
# look carefully for explicit exclusions or negative constraints, such as sentences starting 
# with "do not receive," "are not eligible," "is not obligated," or "program does not apply." 

# Always prioritize specific eligibility criteria (FTE requirements, length of service) 
# and check the FAQ sections provided in the context for clarifications.

# If the context explicitly states that a group does NOT receive a benefit or is NOT eligible, 
# your answer must reflect that.

# Context: {context}
# Question: {question}

# Helpful Answer:"""

custom_prompt_template = """استخدم المعلومات التالية للإجابة على سؤال المستخدم.

إذا كان السؤال يتعلق بأهلية فئة معينة من الموظفين (مثل أعضاء هيئة التدريس أو الإداريين)،
ابحث بعناية عن أي استثناءات أو قيود واضحة مثل:
"لا يحق لهم"، "غير مؤهل"، "لا ينطبق عليهم".

يجب إعطاء الأولوية لشروط الأهلية المحددة مثل نسبة الدوام (FTE) أو مدة الخدمة.

إذا كان السياق يذكر بوضوح أن فئة معينة غير مؤهلة أو لا تحصل على ميزة،
فيجب أن تعكس إجابتك ذلك بوضوح.

 تجاهل أرقام الصفحات الموجودة في الفهارس أو جداول المحتويات.

 يفحص بدقة الفئة الوظيفية (Staff vs Faculty) قبل إعطاء رقم الاستحقاق".

أجب بنفس لغة السؤال الأصلي.


السياق:
{context}

السؤال:
{question}

الإجابة:"""

QA_CHAIN_PROMPT = PromptTemplate(
    template=custom_prompt_template, input_variables=["context", "question"]
)

rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
قم بإعادة صياغة السؤال التالي إلى اللغة العربية الفصحى (MSA).
إذا كان السؤال بالفعل باللغة العربية الفصحى، أعده كما هو.
إذا كان باللهجة المصرية أو فرانكو، قم بتحويله إلى فصحى واضحة.

السؤال:
{question}

السؤال المعاد صياغته:
"""
)

def rewrite_query(llm, question):
    prompt = rewrite_prompt.format(question=question)
    response = llm.invoke(prompt)
    
    # Handle different response formats
    if hasattr(response, "content"):
        return response.content.strip()
    return str(response).strip()

def clean_arabic_text(text):
    # إزالة النقاط المتكررة التي تستخدم في الفهارس (مثل .......)
    text = re.sub(r'\.{2,}', ' ', text)
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    return text

# Streamlit caching for RAG setup
@st.cache_resource
def setup_rag_system():
    # Load and split PDF
    # loader = PyPDFLoader("english_policy.pdf")
    loader = PyMuPDFLoader("arabic_policy.pdf")

    pages = loader.load()

    for doc in pages:
        doc.page_content = clean_arabic_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # Embeddings and vector store
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="Qwen-2.5-32b-Instruct", 
        temperature=0
    )

    rewrite_llm = llm  # reuse same model

    # Return the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    return_source_documents=True
)

    return qa_chain, llm

# Setup RAG chain
qa, rewrite_llm = setup_rag_system()
# Streamlit UI
st.title("RAG HR Chatbot")
user_question = st.text_input("Ask a question about the policy:")

if user_question:
    with st.spinner("Searching the policy..."):
        # response = qa.invoke({"query": user_question})

        # Step 1: Rewrite query to MSA
        rewritten_question = rewrite_query(rewrite_llm, user_question)

        st.write("🔄 Rewritten Question (MSA):")
        st.write(rewritten_question)

        # Step 2: Pass rewritten query to RAG
        response = qa.invoke({"query": rewritten_question})


        st.write("### Answer:")
        st.write(response["result"])

        if "source_documents" in response and response["source_documents"]:
            with st.expander("View Source Citations"):
                source_pages = sorted(list(set([doc.metadata.get('page', 0) + 1 for doc in response["source_documents"]])))
                st.write(f"Information retrieved from PDF pages: **{', '.join(map(str, source_pages))}**")
                
                for i, doc in enumerate(response["source_documents"]):
                    page_num = doc.metadata.get('page', 0) + 1
                    st.markdown(f"**Snippet {i+1} (Page {page_num}):**")
                    st.info(doc.page_content)
        else:
            st.warning("Sources were not found in the response. Ensure 'return_source_documents=True' is set in the chain.")