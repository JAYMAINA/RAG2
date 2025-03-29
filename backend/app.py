from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.chains.question_answering import load_qa_chain
import langdetect
from langdetect import detect
#from langchain_community.chat_models import OpenAI  
from langchain_openai import OpenAI  
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware

import json


load_dotenv()


# Initialize FastAPI
app = FastAPI()

# Define allowed origins (your frontend URL)
origins = [
    "http://localhost:5173",  # Vite Dev Server
    "http://127.0.0.1:5173",
]

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allow only frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize LLM
#llm = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo-instruct")

# Load FAISS knowledge base
def initialize_faiss():
    pdfreader = PdfReader('3.pdf')
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=50,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

document_search = initialize_faiss()

# Language Detection & Translation Functions
def detect_language(text):
    return detect(text)

def translate_to_english(text):
    detected_lang = detect_language(text)
    if detected_lang != "en":
        return GoogleTranslator(source=detected_lang, target="en").translate(text), detected_lang
    return text, "en"

def translate_from_english(text, target_lang):
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    return text

def translate_to_swahili(text, target_lang):
    """Only translate back to Swahili if the original language was not English"""
    if target_lang != "en":
        return GoogleTranslator(source="en", target="sw").translate(text)
    return text

def is_poa_related(query):
    poa_keywords = ['poa', 'internet', 'wifi', 'connection', 'service', 'payment', 'subscription', 'account']
    return any(keyword in query.lower() for keyword in poa_keywords)

# Define Payload Model
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        query = request.question
        translated_query, original_lang = translate_to_english(query)
        
        # Convert question to standalone format
        standalone_question_template = PromptTemplate(
            input_variables=["conv_history", "question"],
            template="""Given some conversation history (if any) and a question, convert the question into a standalone question.
            conversation history: {conv_history}
            question: {question}
            standalone question:"""
        )
        llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
        standalone_question_chain = LLMChain(llm=llm, prompt=standalone_question_template)
        print("Received Question:", query)
        print("Translated Query:", translated_query)

        standalone_question_response = standalone_question_chain.run({
            "conv_history": "",
            "question": translated_query
        })

        if isinstance(standalone_question_response, dict):
            standalone_question = standalone_question_response.get("text", "").strip()
        else:
            standalone_question = str(standalone_question_response).strip()

        print("Standalone Question Response:", standalone_question_response)
        print("Standalone Question:", standalone_question)

        if is_poa_related(standalone_question):
            chain = load_qa_chain(llm, chain_type="stuff")
            docs = document_search.similarity_search(standalone_question)

            print("FAISS Search Docs:", docs)

            response = chain.run({"input_documents": docs, "question": standalone_question})
            print("LLM Chain Response:", response)

            if isinstance(response, dict):
                answer = response.get("text", "").strip()
            else:
                answer = str(response).strip()

            print("FAISS Answer:", answer)

            if original_lang != "en":  
                answer = GoogleTranslator(source="en", target="sw").translate(answer)
        else:
            llm = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo-instruct")
            response = llm.invoke(
                f"You are Lilly, your friendly and helpful AI assistant for poa! internet. "
                f"Respond to the following in a friendly and helpful manner: {standalone_question}"
            )

            print("LLM Raw Response:", response)

            if isinstance(response, dict):
                answer = response.get("text", "").strip()
            else:
                answer = str(response).strip()

        print("Final Answer Before Translation:", answer)

        if original_lang != "en":
            answer = GoogleTranslator(source="en", target="sw").translate(answer)

        print("Final Translated Answer:", answer)

        return {"response": answer}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
