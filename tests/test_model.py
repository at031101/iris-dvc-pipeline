import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

@pytest.fixture
def model():
    return joblib.load("model.pkl")

@pytest.fixture
def testdata():
    return pd.read_csv("data/iris.csv")

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
    assert testdata.shape[1] == 5  # 4 features + 1 target

def test_no_missing_values(testdata):
    assert testdata.isnull().sum().sum() == 0
