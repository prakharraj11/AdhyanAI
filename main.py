from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import os
import sys
import shutil
from rag import QAPipeline, PDFIngester

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "index.html") # this basically fetches the path of the frontend file
UPLOAD_DIR = os.path.join(BASE_DIR, "temp_uploads") # this fetches the path of the place where we are storing the uploaded file

os.makedirs(UPLOAD_DIR, exist_ok=True) # this creates a new directory if it doesn't exist 
sys.path.append(BASE_DIR)

app = FastAPI(title="Adhyan RAG API")

# Setting up middleware for FastAPI to work allowing methods from unkown sources and everything
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#creating pydantic models to check the retrieved information for its correctness and type check
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str

pipeline = None

@app.on_event("startup")
async def startup_event():
    global pipeline    
    # Try to load default pipeline if the default PDF exists
    ingester = PDFIngester()
    vectorstore = ingester.load_vectorstore(ingester.index_path)
    if vectorstore:
        pipeline = QAPipeline(vectorstore=vectorstore)


#defining the first API for the startup process to show the landing page I am just giving a File as a reponse to anyone who reaches this URL
@app.get("/")
async def read_root():
    if os.path.exists(HTML_PATH):
        return FileResponse(HTML_PATH)
    return {"error": "index.html not found"}


#defining the upload endpoint as a post endpoint which will basically help us upload files into the system which is very interesting in its self.
@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global pipeline
    
    if not file.filename.endswith('.pdf'): #not allowing any non pdfs to be entered
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location,"wb") as buffer: 
            shutil.copyfileobj(file.file, buffer) 
        # what we have done in the previous line is very interesting as we have simply opened a new binary file on our computer and file.file is the file that the
        # user has uploaded now what happens here is simply the shutil function pours the file information into the file created on our system also shutilcopyfile obj 
        # creates small bins or buckets in your ram and in this way does not eat you ram during large uploads.
        ingester = PDFIngester()
        vectorstore = ingester.ingest_pdf(file_location)
        pipeline = QAPipeline(vectorstore=vectorstore)
        return {"filename": file.filename, "status": "Processed and ready for chat"}
    except Exception as e:
        print(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

#the post api that takes in the user query and gives out the model response

@app.post("/api/ask", response_model=AnswerResponse) #uses the pydantic response model to type check the model information being sent
async def ask_question(request: QuestionRequest):
    global pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Please upload a PDF document first.")
    
    try:
        answer_text = pipeline.answer_question(request.question)
        return {"answer": answer_text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)