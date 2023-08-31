from fastapi import FastAPI, HTTPException
import uvicorn
from predict import predict_image,calculate_average_entropy_and_histogram
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
app = FastAPI()

from typing import List
@app.post("/predict")
async def predict(image: UploadFile):
    try:
        # Read the uploaded image data
        image_data = await image.read()
        # Convert the image data to a PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Save the uploaded image to a file (e.g., "uploaded_image.jpg")
        image_path = "uploaded_image.jpg"
        pil_image.save(image_path)
        
        average_entropy, avg_histograms=calculate_average_entropy_and_histogram(pil_image)
        prediction = predict_image(pil_image)
        if(prediction):
            return {"prediction": prediction}
        else:
            return HTTPException(status_code=500, detail=str('error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

