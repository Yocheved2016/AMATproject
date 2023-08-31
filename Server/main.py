from fastapi import FastAPI, HTTPException
import uvicorn
from predict import predict_image,calculate_average_entropy_and_histogram,get_distance
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import mlflow

app = FastAPI()


@app.post("/predict")
async def predict(image: UploadFile):
    try:
        mlflow.set_experiment("model_monitoring")
        with mlflow.start_run():
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            image_path = "uploaded_image.jpg"
            pil_image.save(image_path)
            prediction = predict_image(pil_image)
            mlflow.set_tag("class", prediction)
            if prediction:
                entropy, histograms = calculate_average_entropy_and_histogram(pil_image)
                hist_dist = get_distance(pil_image, prediction, histograms)
                metrics = {"Entropy": entropy,
                           "Histogram_distance": hist_dist}
                mlflow.log_metrics(metrics)
                return {"prediction": prediction}

            else:
                return HTTPException(status_code=500, detail=str('error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
