
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Updated import

def process_pdf(pdf_path, pdf_name, embeddings):
    # Extract text from PDF
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Segment text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create and store embeddings
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=pdf_name,
        persist_directory="./chroma_db"
    )
    vector_store.persist()
    
    return True