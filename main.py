import pytesseract
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Textract AI Backend")

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_text_from_image(img):
    """Extract text from an image using Tesseract OCR."""
    text = pytesseract.image_to_string(img)
    return text


@app.post("/api/extract")
async def extract_text(file: UploadFile = File(...)):
    """Accept an image file and return the extracted text."""
    # Read uploaded file
    contents = await file.read()

    # Convert to numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Extract text
    text = get_text_from_image(img)

    return {"text": text, "filename": file.filename}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file."""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)