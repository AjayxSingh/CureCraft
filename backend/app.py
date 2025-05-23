from fastapi import FastAPI
from contextlib import asynccontextmanager
from model.rag_pipeline import load_documents, get_vectorstore, get_rag_chain
from pydantic import BaseModel
import uvicorn
from fastapi.responses import JSONResponse

# Global variable to hold the chain
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain
    # Load everything ONCE during app startup
    documents = load_documents("data/pubmed.json")
    vectorstore = get_vectorstore(documents)
    rag_chain = get_rag_chain(vectorstore)
    yield

# Initialize app with lifespan
app = FastAPI(title="Medical AI Assistant RAG API", lifespan=lifespan)

# Request schema
class QueryRequest(BaseModel):
    content: str

@app.post("/diagnose")
async def diagnose(request: QueryRequest):
    try:
        result = rag_chain.invoke(request.content)
        return {"response": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
