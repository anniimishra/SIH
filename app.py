from fastapi import FastAPI, HTTPException
from PyPDF2 import PdfReader
from pydantic import BaseModel
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os



app = FastAPI()

class InitConversationResponse(BaseModel):
    message: str

class UserInput(BaseModel):
    question: str

class ChatHistoryResponse(BaseModel):
    chat_history: list

conversation_chain = None
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with open(pdf, "rb") as file:
                
                pdf_reader = PdfReader(file)
                
                text += "".join(page.extract_text() for page in pdf_reader.pages)
                
        except Exception as e:
            print(f"Error processing PDF '{pdf}': {str(e)}")
    return text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAFoxSMMXltqhA2iiQ5f--u1tn6oJ4sDRA'

def get_vector_store(text_chunks):
    print(13)
    try:
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise ValueError("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

        embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
        print(14)
    except Exception as e:
        print(f"Error initializing embeddings: {str(e)}")
        raise  # Re-raise the exception to propagate the error

    
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    
    return vector_store




def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return chain

def train_model_from_directory(pdf_folder):
    try:
        pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        vector_store = get_vector_store(text_chunks)
        return get_conversational_chain(vector_store)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/init_conversation", response_model=InitConversationResponse)
async def init_conversation():
    global conversation_chain
    try:
        conversation_chain = train_model_from_directory("train_files")
        print(conversation_chain)
        return {"message": "Conversation Initialized!","chain":conversation_chain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/process_question", response_model=ChatHistoryResponse)
async def process_question(user_input: UserInput):
    global conversation_chain
    try:
        print(1)
        user_question = user_input.question
        response = conversation_chain({'question': user_question})
        chat_history = response['chat_history']
        print(response)
        return {"chat_history": [{"role": message.role, "content": message.content} for message in chat_history]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


