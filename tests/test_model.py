import pytest
import joblib
<<<<<<< HEAD
import pandas as pd
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="module")
def model_and_data():
    model = joblib.load("model.pkl")
    df = pd.read_csv("data/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    return model, X, y

def test_model_accuracy(model_and_data):
    """Ensure model performs above threshold."""
    model, X, y = model_and_data
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc >= 0.8, f"Model accuracy too low: {acc}"

def test_model_predict_shape(model_and_data):
    """Check that model predictions match data length."""
    model, X, _ = model_and_data
    preds = model.predict(X)
    assert len(preds) == len(X), "Prediction length mismatch!"
=======
import os
from sklearn.metrics import accuracy_score

TEST_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture(scope="session")
def model():
    model_path = os.path.join(TEST_DIR, "..", "model.pkl")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    return joblib.load(model_path)

@pytest.fixture(scope="session")
def testdata():
    data_path = os.path.join(TEST_DIR, "data", "iris.csv")
    assert os.path.exists(data_path), f"Data file not found at {data_path}"
    return pd.read_csv(data_path)

def test_model_loads(model):
    assert model is not None

def test_model_predictions(model, testdata):
    X = testdata.drop("target", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(testdata)

def test_model_accuracy(model, testdata):
    X = testdata.drop("target", axis=1)
    y = testdata["target"]
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.90, f"Accuracy {accuracy:.2f} is below threshold"

def test_prediction_range(model, testdata):
    X = testdata.drop("target", axis=1)
    predictions = model.predict(X)
    assert set(predictions).issubset({0, 1, 2})

def test_data_shape(testdata):
    assert testdata.shape[1] == 5

def test_no_missing_values(testdata):
    assert testdata.isnull().sum().sum() == 0
>>>>>>> ee3edf1 (Fix test_model.py to use absolute paths for model.pkl and iris.csv for pytest compatibility)
