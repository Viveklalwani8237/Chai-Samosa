from fastapi import FastAPI, Query
from pydantic import BaseModel
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sklearn

# Confirm scikit-learn version
print("Using scikit-learn version:", sklearn.__version__)

# Path to store and load the model
MODEL_PATH = "./Model/R2.pkl"

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


# Load model function
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file_obj:
            return pickle.load(file_obj)
    return None


# Load existing model if available
iris_model = load_model()

# Create the FastAPI app
app = FastAPI()


# Root route
@app.get("/")
async def root():
    return {"message": "FastAPI Iris Model is running."}


# Request body model for prediction
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Prediction endpoint
@app.post("/predict")
async def predict_iris(features: IrisFeatures):
    global iris_model
    if iris_model is None:
        return {"error": "Model not trained yet. Please train the model first."}

    input_data = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    prediction = iris_model.predict(input_data)
    flower_type = int(prediction[0])
    return {"prediction": flower_type}


# Request body model for training
class TrainParams(BaseModel):
    max_depth: int
    min_samples_leaf: int
    criterion: str  # 'gini' or 'entropy'


# Training endpoint
@app.post("/train")
async def train_model(params: TrainParams):
    global iris_model

    # Load dataset
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    # Train model
    clf = DecisionTreeClassifier(
        max_depth=params.max_depth,
        min_samples_leaf=params.min_samples_leaf,
        criterion=params.criterion
    )
    clf.fit(X_train, y_train)

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    # Reload into memory
    iris_model = clf

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "message": "Model trained and saved successfully.",
        "accuracy": round(accuracy, 4),
        "params": params.model_dump()  # Pydantic v2
    }


# Optional: Get valid hyperparameter options
@app.get("/get-model-params")
async def get_model_params():
    return {
        "max_depth": [4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3],
        "criterion": ["gini", "entropy"]
    }
