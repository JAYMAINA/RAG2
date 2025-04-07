from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langchain.chains.question_answering import load_qa_chain
import langdetect
from langdetect import detect_langs
from langdetect import detect 
from langchain_openai import OpenAI 
from langchain_openai import ChatOpenAI   
from langchain.chains import LLMChain
#from langchain.runnables import RunnableSequence
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime
from db import chat_collection

import json

import os 


load_dotenv()

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# Load FAISS knowledge base
def initialize_faiss():
    embeddings = OpenAIEmbeddings()
    index_path = "faiss_index_react"

    if os.path.exists(index_path):
        print("Loading existing FAISS vector store...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("No existing FAISS index found. Creating new one from PDF...")

    pdfreader = PdfReader('3.pdf')
    raw_text = ''
    for page in pdfreader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    texts = text_splitter.split_text(raw_text)

    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local(index_path)
    print(" FAISS index created and saved locally.")
    return vectorstore


document_search = initialize_faiss()
#TO DO - IMPLEMENT GRAMMAR CORRECTION
# def correct_grammar(text):
#     url = "https://api.languagetool.org/v2/check"
#     params = {
#         'text': text,
#         'language': 'en',
#     }
#     response = requests.post(url, data=params)
#     result = response.json()
    
#     corrected_text = text
#     for match in result['matches']:
#         start, end = match['offset'], match['offset'] + match['length']
#         corrected_text = corrected_text[:start] + match['replacements'][0]['value'] + corrected_text[end:]
    
#     return corrected_text

def detect_language(text):
    try:
        langs = detect_langs(text)
        top_lang = langs[0]
        if top_lang.lang == "en" and top_lang.prob > 0.90:
            return "en"
    except:
        pass
    return "sw"

def translate_to_english(text):
    lang = detect_language(text)
    if lang == "sw":
        return GoogleTranslator(source="sw", target="en").translate(text), "sw"
    return text, "en"

def translate_from_english(text, target_lang):
    if target_lang == "sw":
        return GoogleTranslator(source="en", target="sw").translate(text)
    return text


def ensure_friendly_tone(answer: str) -> str:
    fallback = (
        "I'm here to help! While I may not have the exact answer right now, "
        "you can raise a support ticket at https://customer.poa.im/ under 'Help & Support' "
        "or reach out to our customer care. We're here for you!"
    )
    
    if "i don't know" in answer.lower() or "i am not sure" in answer.lower() or "not mentioned" in answer.lower():
        return fallback

    # Ensure reassurance and empathy
    if not any(phrase in answer.lower() for phrase in ["we're looking into it", "we're here to help", "please contact", "you're not alone"]):
        answer += " If the issue persists, don't worry—our team is always here to help you out!"

    return answer


def is_poa_related(query):
    poa_keywords = ['poa', 'internet', 'wifi', 'connection', 'service', 'payment', 'subscription', 'account']
    return any(keyword in query.lower() for keyword in poa_keywords)

import re

def prefix_with_poa(text: str) -> str:
    keywords = ['internet', 'wifi', 'connection', 'service', 'payment', 'subscription', 'account','litebeam','haplite','router','dish','installation','speeds']
    for word in keywords:
        # \b ensures we only match whole words, not substrings
        pattern = rf'\b{word}\b'
        text = re.sub(pattern, f'poa! {word}', text, flags=re.IGNORECASE)
    return text


# Define Payload Model
class QuestionRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = []
    session_id: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        query = request.question
        translated_query, original_lang = translate_to_english(query)
        print("Original Language is ",original_lang)
        
        
        # Convert question to standalone format
        standalone_question_template = PromptTemplate(
            input_variables=["conv_history", "question"],
            template="""Given some conversation history (if any) and a question, convert the question into a standalone question.
            conversation history: {conv_history}
            question: {question}
            standalone question:"""
        )
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        standalone_question_chain = standalone_question_template | llm
        print("Received Question:", query)
        print("Translated Query:", translated_query)
        print ("StandAlone Question:",standalone_question_template)

     # Build conversation history string for context
        formatted_history = ""
        for msg in request.history[-10:]:  # Limit to last 10 exchanges
            role = msg["role"]
            content = msg["content"]
            formatted_history += f"{role}: {content}\n"

        standalone_question_response = standalone_question_chain.invoke({
            "conv_history": formatted_history,
            "question": translated_query
        })

        if isinstance(standalone_question_response, dict):
            standalone_question = standalone_question_response.get("text", "").strip()
        else:
            standalone_question = str(standalone_question_response).strip()

        print("Standalone Question Response:", standalone_question_response)
        print("Standalone Question:", standalone_question)
        
        standalone_question = prefix_with_poa(standalone_question)
        print("Poa! restructered Standalone Question:", standalone_question)

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
            
            answer = ensure_friendly_tone(answer)


            print("FAISS Answer:", answer)

            # if original_lang == "sw":  
            #     answer = GoogleTranslator(source="en", target="sw").translate(answer)
        else:
            llm = ChatOpenAI(temperature=0.8, model_name="gpt-4o-mini")
            response = llm.invoke(
                f"""You are Lilly, a warm and supportive AI assistant for poa! internet. 
                Respond in a friendly, empathetic, and helpful tone. Always reassure the user 
                that their concern is valid and being looked into. Never say you don't know. 
                Speak in first or second person, and if unsure, guide the user gently to contact support.

                Here is the question: {standalone_question}"""                               
            )

            print("LLM Raw Response:", response)
            

            if isinstance(response, dict):
                answer = response.get("text", "").strip()
            else:
                answer = str(response).strip()
                
            answer = response.content if hasattr(response, 'content') else str(response)
            answer = ensure_friendly_tone(answer)

        print("Final Answer Before Translation:", answer)

        # if original_lang == "sw":
        #     answer = GoogleTranslator(source="en", target="sw").translate(answer)

        print("Final Translated Answer:", answer)
        
        await chat_collection.update_one(
                {"session_id": request.session_id},
                {
                    "$push": {
                        "history": {
                            "$each": request.history + [{"role": "assistant", "content": answer}]
                        }
                    },
                    "$setOnInsert": {"timestamp": datetime.utcnow()}
                },
                upsert=True
            )

        return {"response": answer}
    
       

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



#TO DO 1.CONTEXTUALIZATION 2. LOCAL EMBEDINGS UNLESS PDF FILE IS CHANGED 3. DATA STORAGE 4.UPDATE DATA SOURCE FROM RESPONSE 5. ENSURE RESPONSES ARE ALWAYS FRIENDLY