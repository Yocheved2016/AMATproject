from fastapi import FastAPI, HTTPException
import uvicorn
from Server.predict import predict_image, calculate_average_entropy_and_histogram
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
            print('sending to function')
            try:
                entropy, histograms = calculate_average_entropy_and_histogram(pil_image)
                print(f'back form function results: entropy: {entropy}, histograms: {histograms}')
            except Exception as e:
                print(f'function fell into error: {e}')
            # mlflow.log_histogram
            hist_dist = 0
            metrics = {"Entropy": entropy,
                       "Histogram_distance": hist_dist,
                       # "red histogram": histograms[0].tolist(),
                       # "green histogram": histograms[1].tolist(),
                       # "blue histogram": histograms[2].tolist()
                       }
            mlflow.log_metrics(metrics)
            prediction = predict_image(pil_image)
            mlflow.set_tag("class", prediction)
            if (prediction):
                return {"prediction": prediction}
            else:
                return HTTPException(status_code=500, detail=str('error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
