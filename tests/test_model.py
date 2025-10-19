import pytest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

@pytest.fixture(scope="session")
def model():
    """Load the trained model from file."""
    return joblib.load("model.pkl")

@pytest.fixture(scope="session")
def testdata():
    """Load the test dataset."""
    return pd.read_csv("data/iris.csv")

def test_model_loads(model):
    """Ensure the model loads successfully."""
    assert model is not None

def test_model_predictions(model, testdata):
    """Check prediction count matches input data rows."""
    X = testdata.drop("target", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(testdata)

def test_model_accuracy(model, testdata):
    """Ensure model accuracy exceeds threshold."""
    X = testdata.drop("target", axis=1)
    y = testdata["target"]
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    assert accuracy > 0.90, f"Accuracy {accuracy:.2f} is below threshold"

@pytest.mark.parametrize("expected_label_set", [{0, 1, 2}])
def test_prediction_range(model, testdata, expected_label_set):
    """Ensure predictions fall within expected class labels."""
    X = testdata.drop("target", axis=1)
    predictions = model.predict(X)
    assert set(predictions).issubset(expected_label_set)

def test_data_shape(testdata):
    """Validate the dataset has expected number of columns (4 features + 1 target)."""
    assert testdata.shape[1] == 5

def test_no_missing_values(testdata):
    """Verify there are no missing values in the dataset."""
    assert testdata.isnull().sum().sum() == 0
