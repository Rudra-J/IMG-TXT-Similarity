from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from app.api.routes import router

app = FastAPI(title="Multimodal Document Similarity")
app.include_router(router)
