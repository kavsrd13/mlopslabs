# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Model import IrisModel, IrisSpecies

# 2. Create app and model objects
app = FastAPI()
model = IrisModel()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Prediction API! Use the /predict endpoint to make predictions."}


@app.post('/predict')
def predict_species(iris: IrisSpecies):
    try:
        prediction, probability = model.predict_species(
            iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width
        )
        return {
            'prediction': prediction,
            'probability': probability
        }
    except Exception as e:
        return {"error": str(e)}

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000)