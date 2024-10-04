import plotly.express as px
import numpy as np


def plot_activation_distribution(activations: np.ndarray, feature_index: int, hook_name: str):
    data = activations[:, feature_index]
    fig = px.histogram(data, nbins=50, title=f"Activation Distribution for Feature {feature_index} in {hook_name}")
    fig.show()

def plot_probe_accuracy(history: dict):
    fig = px.line(x=list(history.keys()), y=list(history.values()), labels={'x': 'Probe', 'y': 'Accuracy'}, title="Probe Training Accuracy")
    fig.show()

def visualize_model_weights(model_data, hook_name: str):
    weights = model_data.coef_
    if weights.ndim > 1:
        weights = weights[0]  # Taking the first class for binary classification
    fig = px.bar(x=list(range(len(weights))), y=weights, labels={'x': 'Feature Index', 'y': 'Weight Value'}, title=f"Linear Probe Weights for {hook_name}")
    fig.show()
