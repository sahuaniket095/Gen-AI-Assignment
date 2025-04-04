from pydantic import BaseModel

class PDFInput(BaseModel):
    pdf_name: str

class QuestionInput(BaseModel):
    question: str
    pdf_name: str