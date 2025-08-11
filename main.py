from fastapi import FastAPI, File, UploadFile
import zipfile, os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

UPLOAD_DIR = "uploaded_data"
templates = Jinja2Templates(directory="templates")

@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Unzip
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)

    return {"message": "File uploaded and extracted"}

def get_file_structure(path):
    tree = {}
    for root, dirs, files in os.walk(path):
        tree[root] = {"dirs": dirs, "files": files}
    return tree

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    structure = get_file_structure(UPLOAD_DIR)
    return templates.TemplateResponse("index.html", {"request": request, "structure": structure})
