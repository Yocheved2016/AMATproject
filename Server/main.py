from fastapi import FastAPI, HTTPException
import uvicorn
from predict import predict_image,calculate_average_entropy_and_histogram,get_distance, class_names
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import mlflow

app = FastAPI()


@app.post("/predict")
async def predict(image: UploadFile):
    try:
        mlflow.set_experiment("cifar_model_monitoring")
        with mlflow.start_run():
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            image_path = "uploaded_image.jpg"
            pil_image.save(image_path)
            prediction, confidence = predict_image(pil_image)
            print(f'prediction: {prediction}, confidence: {confidence}')
            if prediction is not None:
                try:
                    entropy, histograms = calculate_average_entropy_and_histogram(pil_image)
                    hist_dist = get_distance(pil_image, prediction, histograms)
                    metrics = {"Entropy": entropy,
                               "Histogram_distance": hist_dist,
                               "Confidence": confidence}
                    mlflow.log_metrics(metrics)
                    mlflow.set_tag("class", class_names[prediction])
                    return class_names[prediction]
                except Exception as e:
                    print(e)
            else:
                return HTTPException(status_code=500, detail=str('error'))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/incorrectPrediction")
async def incorrect_prediction(image: UploadFile,correct_class):
    image_data = await image.read()
    # Convert the image data to a PIL Image
    pil_image = Image.open(io.BytesIO(image_data))

    # Generate a random string 
    random_uuid = uuid.uuid4()
    # Save the uploaded image to a folder
    image_path = f'Candidates/{correct_class}/{random_uuid}.jpg'
    pil_image.save(image_path)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
