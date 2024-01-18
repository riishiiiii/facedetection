import face_recognition
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO

app = FastAPI()


@app.post("/compare_faces")
async def compare_faces_endpoint(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    try:
        result = compare_faces(await image1.read(), await image2.read())

        return JSONResponse(content={"result": result})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def compare_faces(img1_content, img2_content):
    try:
        img1 = face_recognition.load_image_file(BytesIO(img1_content))
        img2 = face_recognition.load_image_file(BytesIO(img2_content))

        face_locations_img1 = face_recognition.face_locations(img1)
        face_locations_img2 = face_recognition.face_locations(img2)

        if not face_locations_img1 or not face_locations_img2:
            return "Face not found"

        face_encodings_img1 = face_recognition.face_encodings(img1, face_locations_img1)
        face_encodings_img2 = face_recognition.face_encodings(img2, face_locations_img2)

        face_encodings_img1 = np.array(face_encodings_img1)
        face_encodings_img2 = np.array(face_encodings_img2)

        results = face_recognition.compare_faces(face_encodings_img1, face_encodings_img2)

        if True in results:
            return True
        else:
            return False
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
        uvicorn.run(
        "main:app",
        port=8001
    )
