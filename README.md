# ProbeLens

**ProbeLens** is a **barebones** linear probe toolkit designed for [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and [SAELens](https://github.com/jbloomAus/SAELens). It facilitates capturing activations from specified hook points, training linear probes using scikit-learn, logging metrics with Weights and Biases (W&B), and visualizing results with Plotly.

## Features

- **Activation Capture:** Attach hooks to `HookedTransformer` and `SAE` models to capture and store activations.
- **Linear Probe Training:** Train separate linear probes for each hook using scikit-learn.
- **Weights and Biases Integration:** Log training metrics for easy monitoring and comparison.
- **Visualization:** Utilize Plotly to visualize activation distributions and probe weights.

## Installation

Install ProbeLens via pip:

```bash
pip install git+https://github.com/tom-pollak/ProbeLens.git
```

## Usage

### 1.

```python
```

### 2. Training Linear Probes

```python
```

### 3. Visualizing Results

```python
```

### Logging with Weights and Biases

Ensure you've set up W&B and logged in:

```bash
wandb login
```

ProbeLens automatically logs training metrics such as accuracy for each probe. View them in your W&B dashboard under the specified project name.


