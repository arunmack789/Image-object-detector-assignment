from typing import List, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from io import BytesIO
from PIL import Image, ImageDraw
from ultralytics import YOLO
import json
import zipfile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolo11m.pt")

# This will hold the detections in Memory 
detection_results = []
processed_images = {} # dictionary to store processed images

@app.post("/detect")
async def detect_images(files: Union[UploadFile, List[UploadFile]] = File(...)):
    global detection_results
    global processed_images
    
    # Normalize input to list
    if not isinstance(files, list):
        files = [files]

    new_detections = []
    
    try:
        for file in files:
            print(f"Processing file: {file.filename}")
            image_bytes = await file.read()
            img = Image.open(BytesIO(image_bytes))

            # Perform detection with YOLO
            results_list = model(img)
            result = results_list[0]
            
            # Draw bounding boxes and labels on the image
            draw = ImageDraw.Draw(img)
            detections_json = []

            for box in result.boxes:
                # Extract coordinates, confidence, and class
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                
                # Draw rectangle
                draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                draw.text((x_min, y_min - 10), label, fill="orange")

                # Store detection info in JSON format
                detections_json.append({
                    "box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    },
                    "confidence": confidence,
                    "class": class_name
                })
            
            # Save the processed image to memory
            processed_image_bytes_io = BytesIO()
            img.save(processed_image_bytes_io, format="PNG")
            processed_images[file.filename] = processed_image_bytes_io.getvalue()
            
            # Store the detection results
            new_detection = {"filename": file.filename, "detections": detections_json}
            new_detections.append(new_detection)

        detection_results.extend(new_detections)

        # Return the JSON response as before
        return {"all_detections": detection_results}

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/reset")
async def reset_results():
    """Endpoint to clear the in-memory detections and images."""
    global detection_results
    global processed_images
    detection_results = []
    processed_images = {}
    return {"message": "In-memory detections and images have been reset."}

@app.get("/download")
async def download_results():
    """Endpoint to download the in-memory results as a JSON file."""
    global detection_results
    if detection_results:
        headers = {'Content-Disposition': 'attachment; filename="detection_results.json"'}
        return JSONResponse(content=detection_results, headers=headers)
    return JSONResponse(status_code=404, content={"message": "No detection results to download."})

@app.get("/images/{filename}")
async def get_processed_image(filename: str):
    """Endpoint to serve a processed image."""
    global processed_images
    image_bytes = processed_images.get(filename)
    if image_bytes:
        return Response(content=image_bytes, media_type="image/png")
    return Response(status_code=404, content="Image not found")

@app.get("/download_zip")
async def download_zip():
    """Endpoint to download all processed images and a JSON file in a single zip."""
    global detection_results
    global processed_images

    if not detection_results:
        return JSONResponse(status_code=404, content={"message": "No detection results to zip."})

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add a separate JSON file for each image
        for detection in detection_results:
            filename = detection['filename'].split('.')[0]
            json_filename = f"{filename}.json"
            json_data = json.dumps(detection, indent=2)
            zip_file.writestr(f"output_json/{json_filename}", json_data)

        # Add all processed images
        for filename, image_bytes in processed_images.items():
            zip_file.writestr(f"output_images/{filename}", image_bytes)
            
    zip_buffer.seek(0)
    
    headers = {
        "Content-Disposition": "attachment; filename=detection_results.zip",
        "Content-Type": "application/zip",
    }
    return StreamingResponse(zip_buffer, headers=headers)
