from typing import List
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
import httpx
import os

app = FastAPI()

AI_BACKEND = os.getenv("AI_BACKEND_URL", "http://ai-backend:8000")
AI_BACKEND_PUBLIC = os.getenv("AI_BACKEND_PUBLIC_URL", "http://localhost:8000")

# This endpoint serves the HTML form for uploading files.
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Detection</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ccc; border-radius: 8px; }}
            h1 {{ text-align: center; }}
            pre {{ background-color: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }}
            button {{ padding: 10px; margin-top: 10px; }}
            input[type="file"] {{ margin-bottom: 10px; }}
            .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }}
            .image-box {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; text-align: center; }}
            .image-box img {{ max-width: 300px; height: auto; }}
        </style>
        <script>
            // This script resets the backend's JSON file when the page loads
            window.onload = function() {{
                fetch("{AI_BACKEND_PUBLIC}/reset", {{ method: 'GET' }})
                    .then(response => response.json())
                    .then(data => console.log(data.message))
                    .catch(error => console.error('Error resetting file:', error));
            }};
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Upload Images for Detection</h1>
            <form action="/upload_and_detect/" method="post" enctype="multipart/form-data">
                <label for="files">Select images to upload:</label>
                <input type="file" id="files" name="files" accept="image/*" multiple required>
                <button type="submit">Upload and Detect</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# This endpoint handles the file upload and sends each file to the backend.
@app.post("/upload_and_detect/", response_class=HTMLResponse)
async def upload_and_detect(files: List[UploadFile] = File(...)):
    if not files:
        return HTMLResponse("<h1>No files received</h1>")

    reset_url = f"{AI_BACKEND}/reset"
    detect_url = f"{AI_BACKEND}/detect"
    download_zip_url = f"{AI_BACKEND_PUBLIC}/download_zip"

    try:
        # Reset
        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.get(reset_url)

        # Send files
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(detect_url, files=[('files', (file.filename, await file.read(), file.content_type)) for file in files])
            final_response = response.json()

        html_output = "<h1>Detection Results</h1>"
        html_output += "<p>Processing complete. Your results are ready for download.</p>"
        html_output += f"<a href='{download_zip_url}'><button>Download All Results (ZIP)</button></a>"

        return HTMLResponse(content=html_output)

    except httpx.RequestError as exc:
        return HTMLResponse(f"<h1>An error occurred while requesting {exc.request.url!r}.</h1>")
