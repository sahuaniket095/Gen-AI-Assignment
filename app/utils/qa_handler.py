from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

def get_answer(question, pdf_name, embeddings, llm):
    vector_store = Chroma(
        collection_name=pdf_name,
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "source_context": [doc.page_content for doc in result["source_documents"]]
    }