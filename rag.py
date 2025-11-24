import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()  
googleAPI= os.getenv("GOOGLE_API_KEY")

# Configiuring the Google API key
api_key = googleAPI
genai.configure(api_key=api_key)

class PDFIngester:
    def __init__(self, embedding_model: str = "models/gemini-embedding-001", chunk_size: int = 1000, chunk_overlap: int = 200):
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key  
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Path in the latest commit
        base_path = os.getcwd()
        self.index_path = os.path.join(base_path, "vectors", "faiss_index")
        
    def extract_text(self, pdf_path: str) -> List[Document]:
        reader = PdfReader(pdf_path)
        documents = []
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text.strip():  
                doc = Document(
                    page_content=text.strip(),
                    metadata={"source": pdf_path, "page": i}
                )
                documents.append(doc)
        return documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(documents)
        return chunks

    def ingest_pdf(self, pdf_path: str, save_index: bool = True) -> FAISS:
        try:
            docs = self.extract_text(pdf_path)
            if not docs:
                raise ValueError("No text extracted from PDF. It might be scanned—consider OCR.")

            chunks = self.chunk_documents(docs)
            if not chunks:
                raise ValueError("No chunks created. Check PDF content.")

            vectorstore = FAISS.from_documents(chunks, self.embeddings)

            if save_index:
                # usng the dynamic path for directory access and creation
                folder_path = os.path.dirname(self.index_path)
                os.makedirs(folder_path, exist_ok=True)
                vectorstore.save_local(self.index_path)

            return vectorstore

        except Exception as e:
            print(f"Ingestion failed: {str(e)}")
            raise

    @classmethod
    def load_vectorstore(cls, index_path: str = None, api_key: str = None) -> Optional[FAISS]:
        index_path = index_path
        if os.path.exists(index_path):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key or googleAPI
            )
            return FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        print("No existing index found")
        return None


class QAPipeline:
    def __init__(self, vectorstore: FAISS, embedding_model: str = "models/gemini-embedding-001",
                 llm_model: str = "gemini-2.5-pro", temperature: float = 0.1, top_k: int = 3):

        self.vectorstore = vectorstore
        self.top_k = top_k

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key
        )

        self.llm = ChatGoogleGenerativeAI(
            model=llm_model,
            temperature=temperature,
            google_api_key=api_key
        )
        # best segement defining the syntax of the I/O with the model
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based solely on the provided context from a PDF document.
            If the answer isn't in the context, say "I couldn't find that information in the document."
            Keep responses, accurate, and cite page numbers where possible. When the user asks for more information regarding the pdf in question which is not there in the pdf but relevant
            try to help him out by giving relevvant information from the internet. When the user asks for "CITATIONS" give him links of relevant research papers that are in context of the 
            pdf provded by the user give 5 links and a short description regarding each.""" ),
            ("human", """Context:
               {context}

               Question: {question}

               Answer:""")
        ])

        # Simplified chain: Just prompt | LLM | Parser (retrieval manual below)
        self.chain = self.prompt | self.llm | StrOutputParser()

        print(f"QA Pipeline initialized with LLM={llm_model}, top_k={top_k}")

    def format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "Unknown")
            snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            formatted.append(f"[Chunk {i}, Page {page}]: {snippet}")
        return "\n\n".join(formatted)

    def answer_question(self, question: str) -> str:
        try:
            print(f"Answering question: {question}")

            print("Embedding query...")
            query_embedding = self.embeddings.embed_query(question)
            print("Query embedded successfully!")
            docs = self.vectorstore.similarity_search_by_vector(query_embedding, k=self.top_k)
            context = self.format_docs(docs)
            response = self.chain.invoke({"context": context, "question": question})
            print("Answer generated successfully!")
            return response

        except Exception as e:
            print(f"QA failed: {str(e)}")
            if "metadata.google.internal" in str(e) or "503" in str(e):
                print("ADC auth timeout—manual embedding should prevent this. Check API key quotas.")
            return "Sorry, an error occurred while processing your question."


