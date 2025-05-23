import os
from dotenv import load_dotenv
from langchain.document_loaders import JSONLoader
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

# Load documents
def custom_metadata(record, content):
    return {
        "title": record.get("title"),
        "interventions": record.get("interventions", []),
        "SO": record.get("metadata", {}).get("SO", "")
    }

def load_documents(json_path):
    loader = JSONLoader(
        file_path=json_path,
        jq_schema=".[]",
        content_key="content",
        metadata_func=custom_metadata
    )
    return loader.load()

# Initialize embedding model (local)
embedding_model = HuggingFaceEmbeddings(
    model_name="model/embedding_llm",  
    model_kwargs={"device": "cpu"},
)

# Vectorstore setup
VECTORSTORE_PATH = "vectorstore/pubmed_faiss_biobert"

def get_vectorstore(documents=None):
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    if documents is not None:
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore
    raise ValueError("Documents not provided and vectorstore doesn't exist.")

# Load local LLM
LLM_MODEL_ID = "model/finetuned_llm"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful and knowledgeable AI medical assistant. A patient has reported the following symptoms:

Patient Symptoms:
{query}

Relevant Research or Medical Context:
{context}

Based on the provided symptoms and research context, provide:

1. **Probable Diagnosis** – Mention the most likely condition(s) based on symptoms.
2. **Suggested Treatment Plan** – Include lifestyle changes, medications, or further diagnostic tests.
3. **Cited Medical Sources** – If any research or credible sources were referenced (from 'SO'), mention them briefly.

Ensure the output is written in clear, layman-friendly language, but medically sound.
""")

def format_docs(retrieved_docs):
    return "\n\n".join(
        doc.page_content + ' Interventions: ' + ', '.join(doc.metadata.get('interventions', []))
        for doc in retrieved_docs
    )

def get_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'query': RunnablePassthrough()
    })
    parser = StrOutputParser()
    return parallel_chain | prompt | llm | parser
