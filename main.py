import os
import zipfile
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
from typing import Optional
import json

# Config
UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)
HF_API_KEY = os.getenv("HF_API_KEY")  
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def get_file_structure(root_dir):
    structure = {}
    for dirpath, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        structure[rel_path] = {"dirs": dirs, "files": files}
    return structure


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())

    folder_name = os.path.splitext(file.filename)[0]
    extract_path = os.path.join(UPLOAD_DIR, folder_name)
    os.makedirs(extract_path, exist_ok=True)
    unzip_file(save_path, extract_path)

    return {"message": "File uploaded and extracted successfully", "folder": folder_name}


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    structure = get_file_structure(UPLOAD_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "structure": structure})


@app.get("/search")
def search_dataset(q: Optional[str]):
    if not q:
        return {"error": "Please provide a query ?q=..."}
    # Make a simple prompt for LLM
    prompt = f"The dataset file structure is:\n{json.dumps(get_file_structure(UPLOAD_DIR), indent=2)}\n\nQuestion: {q}\nAnswer:"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    resp = requests.post(
        "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
        headers=headers,
        json={"inputs": prompt}
    )
    try:
        answer = resp.json()[0]["generated_text"]
    except:
        answer = resp.text
    return {"query": q, "answer": answer}
