# Import necessary libraries
from flask import Flask, request, jsonify
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__)

# Define the path for the models directory relative to this script
MODELS_DIR = "./model" # This correctly points to your 'model' folder
# Define the path for the default model file
DEFAULT_MODEL_FILENAME = "iris_classifier.pkl"
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)
# Define the path for the custom trained model file
CUSTOM_MODEL_FILENAME = "my-model.pkl"
CUSTOM_MODEL_PATH = os.path.join(MODELS_DIR, CUSTOM_MODEL_FILENAME)

# --- Start of the application logic ---

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Load the initial iris model or train a default one if not found
iris_model = None
try:
    if os.path.exists(DEFAULT_MODEL_PATH):
        with open(DEFAULT_MODEL_PATH, "rb") as fileObj:
            iris_model = pickle.load(fileObj)
        print(f"Successfully loaded initial model from {DEFAULT_MODEL_PATH}")
    else:
        print(f"Initial model not found at {DEFAULT_MODEL_PATH}. Training a default model...")
        # Load the Iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        # Train a default Decision Tree Classifier
        default_dt_model = DecisionTreeClassifier(random_state=42)
        default_dt_model.fit(X, y)
        # Save the default model
        with open(DEFAULT_MODEL_PATH, "wb") as fileObj:
            pickle.dump(default_dt_model, fileObj)
        iris_model = default_dt_model
        print("Default model trained and saved.")
except Exception as e:
    print(f"Error initializing default model: {e}")
    # In a real application, you might want to exit or log a critical error here.

# --- Flask Routes (Endpoints) ---

@app.route("/", methods=["GET"])
def home():
    """
    Home endpoint for the Flask application.
    Returns a welcome message.
    """
    return "Welcome to the JK Flask ML App! - initial get to test flask is running or not!!"
@app.route("/training", methods=["POST"])
def train_model():
    """
    API to train a new Decision Tree model with provided hyperparameters.
    The trained model is saved to the 'model' directory.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract hyperparameters from the request body
    max_depth = data.get("max_depth")
    min_samples_leaf = data.get("min_samples_leaf")
    criterion = data.get("criterion", "gini") # Default to 'gini' if not provided

    # Validate criterion
    if criterion not in ["gini", "entropy", "log_loss"]:
        return jsonify({"error": "Invalid criterion. Must be 'gini', 'entropy', or 'log_loss'."}), 400

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data into training and testing sets (optional, but good practice for evaluation)
    # For this exercise, we will train on the full dataset for simplicity,
    # as evaluation is not explicitly requested yet.
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        # Initialize Decision Tree Classifier with provided hyperparameters
        # Ensure max_depth and min_samples_leaf are integers if provided
        dt_model = DecisionTreeClassifier(
            max_depth=int(max_depth) if max_depth is not None else None,
            min_samples_leaf=int(min_samples_leaf) if min_samples_leaf is not None else 1,
            criterion=criterion,
            random_state=42 # For reproducibility
        )

        # Train the model on the full Iris dataset
        dt_model.fit(X, y)

        # Save the trained model
        with open(CUSTOM_MODEL_PATH, "wb") as fileObj:
            pickle.dump(dt_model, fileObj)

        return jsonify({"message": "Model trained successfully and saved.", "model_path": CUSTOM_MODEL_PATH}), 200
    except Exception as e:
        return jsonify({"error": f"Error during model training or saving: {str(e)}"}), 500
@app.route("/get-model-param", methods=["GET"])
def get_model_parameters():
    """
    API to get the parameters of the currently active prediction model.
    It first checks for a custom trained model, then defaults to the initial model.
    """
    current_model = None
    # Check if a custom model exists and load it
    if os.path.exists(CUSTOM_MODEL_PATH):
        try:
            with open(CUSTOM_MODEL_PATH, "rb") as fileObj:
                current_model = pickle.load(fileObj)
            print(f"Using custom model from {CUSTOM_MODEL_PATH} for parameters.")
        except Exception as e:
            print(f"Error loading custom model for parameters: {e}")
            current_model = iris_model # Fallback to initial model
    else:
        current_model = iris_model # Use initial model if no custom model

    if current_model:
        # Get parameters of the Decision Tree Classifier
        # The get_params() method returns a dictionary of all parameters
        params = current_model.get_params()
        # Filter for relevant parameters as per the diagram (depth, min_leaf, criterion)
        model_params = {
            "max_depth": params.get("max_depth"),
            "min_samples_leaf": params.get("min_samples_leaf"),
            "criterion": params.get("criterion")
        }
        return jsonify(model_params)
    else:
        return jsonify({"error": "No model loaded or available."}), 500
@app.route("/predict", methods=["POST"])
def iris_prediction():
    """
    API to predict the flower type using the currently active model.
    Prioritizes a custom trained model if available, otherwise uses the initial model.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    # Extract and validate features
    required_params = ["sl", "pl", "sw", "pw"]
    features_input = [data.get(p) for p in required_params]

    if any(x is None for x in features_input):
        return jsonify({"error": f"Missing one or more required parameters: {', '.join(required_params)}"}), 400

    try:
        features = [[float(x) for x in features_input]]
    except ValueError:
        return jsonify({"error": "Invalid input values. Please ensure all measurements are numbers."}), 400

    # Determine which model to use (custom-trained or default)
    current_model = None
    if os.path.exists(CUSTOM_MODEL_PATH):
        try:
            with open(CUSTOM_MODEL_PATH, "rb") as fileObj:
                current_model = pickle.load(fileObj)
            print(f"Using custom model from {CUSTOM_MODEL_PATH} for prediction.")
        except Exception as e:
            print(f"Error loading custom model for prediction: {e}. Falling back to initial model.")
            current_model = iris_model
    else:
        current_model = iris_model

    if not current_model:
        return jsonify({"error": "No model available for prediction."}), 500

    try:
        # Make prediction
        flower_type_index = current_model.predict(features)[0]
        iris_target_names = load_iris().target_names
        predicted_flower_name = iris_target_names[flower_type_index]
        return jsonify({"predicted_flower_type": predicted_flower_name})
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500
# --- Run the Flask app ---
if __name__ == "__main__":
    app.run(debug=True) # Run the app in debug mode for development