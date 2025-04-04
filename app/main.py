from fastapi import FastAPI, UploadFile, File
from .models import PDFInput, QuestionInput
from .utils.pdf_processor import process_pdf
from .utils.qa_handler import get_answer
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings  
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=google_api_key
)
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=google_api_key
)

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), pdf_name: str = None):
    try:
        pdf_path = f"temp_{pdf_name}.pdf"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        
        process_pdf(pdf_path, pdf_name, embeddings)
        os.remove(pdf_path)
        
        return {"message": f"PDF {pdf_name} successfully processed and stored"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask_question/")
async def ask_question(input: QuestionInput):
    try:
        result = get_answer(input.question, input.pdf_name, embeddings, llm)
        return result
    except Exception as e:
        return {"error": str(e)}