import os
import json
import zipfile
import shutil
from typing import Optional, Dict, List

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()


UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = os.getenv("HF_MODEL", "microsoft/DialoGPT-small")



app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def unzip_file(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


def get_file_structure(root_dir: str) -> Dict[str, Dict[str, List[str]]]:
    structure: Dict[str, Dict[str, List[str]]] = {}
    for dirpath, dirs, files in os.walk(root_dir):
        rel_path = os.path.relpath(dirpath, root_dir)
        structure[rel_path] = {"dirs": sorted(dirs), "files": sorted(files)}
    return structure


def is_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"
    }


def compute_stats(root_dir: str):
    stats = {
        "total_files": 0,
        "total_images": 0,
        "splits": {"train": 0, "val": 0, "test": 0},
        "classes": {},
    }
    # global counts
    for _, _, files in os.walk(root_dir):
        stats["total_files"] += len(files)
        stats["total_images"] += sum(1 for f in files if is_image(f))

    # split counts
    for split in ["train", "val", "validation", "test"]:
        p = os.path.join(root_dir, split)
        if os.path.isdir(p):
            c = 0
            for _, _, files in os.walk(p):
                c += sum(1 for f in files if is_image(f))
            if split == "validation":
                stats["splits"]["val"] += c
            else:
                stats["splits"][split] = c

    # class counts under train/*
    train = os.path.join(root_dir, "train")
    if os.path.isdir(train):
        for cls in sorted(os.listdir(train)):
            cls_dir = os.path.join(train, cls)
            if os.path.isdir(cls_dir):
                cnt = 0
                for _, _, files in os.walk(cls_dir):
                    cnt += sum(1 for f in files if is_image(f))
                stats["classes"][cls] = cnt
    return stats


def build_prompt(structure, stats, question: str) -> str:
    top_dirs = sorted(structure.get(".", {}).get("dirs", []))
    top_files = sorted(structure.get(".", {}).get("files", []))
    classes = stats.get("classes", {})
    class_lines = []
    for i, (k, v) in enumerate(sorted(classes.items(), key=lambda x: (-x[1], x[0]))):
        if i >= 50:
            class_lines.append(f"... and {len(classes)-50} more")
            break
        class_lines.append(f"{k}: {v}")

    return (
        "You are a dataset assistant. Use only the provided structure and stats.\n"
        f"Top-level dirs: {top_dirs}\n"
        f"Top-level files: {top_files}\n"
        f"Stats: total_files={stats['total_files']}, total_images={stats['total_images']}, splits={stats['splits']}\n"
        f"Classes(train): {', '.join(class_lines) if class_lines else 'none'}\n\n"
        f"Question: {question}\n"
        "Answer concisely with exact counts when applicable."
    )


def call_hf_inference(prompt: str) -> str:
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HF_API_KEY not set")
    
    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_API_KEY
        )
        
        # For simple text generation
        result = client.text_generation(
            prompt=prompt,
            model=HF_MODEL,
            max_new_tokens=256
        )
        
        return result
        
    except Exception as e:
        return f"Inference failed: {str(e)}"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    structure = get_file_structure(UPLOAD_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "structure": structure})


@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload a .zip file")
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    folder = os.path.splitext(file.filename)[0]
    extract_path = os.path.join(UPLOAD_DIR, folder)
    os.makedirs(extract_path, exist_ok=True)
    unzip_file(save_path, extract_path)
    return {"message": "Uploaded and extracted", "folder": folder}


@app.get("/structure")
def structure():
    return get_file_structure(UPLOAD_DIR)


@app.get("/stats")
def stats():
    return compute_stats(UPLOAD_DIR)


@app.get("/search")
def search(q: str = Query(..., description="LLM question about the dataset")):
    structure = get_file_structure(UPLOAD_DIR)
    stats = compute_stats(UPLOAD_DIR)
    prompt = build_prompt(structure, stats, q)
    answer = call_hf_inference(prompt)
    return {"query": q, "answer": answer}
