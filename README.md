# Neural Network from Scratch (NumPy)

A 2-layer neural network built entirely from scratch in NumPy — no ML frameworks, no autodiff. Every forward pass, gradient, and parameter update is implemented manually, with the backpropagation formulas derived by hand using the chain rule.(Note: Used cluade to write this report , but the whole code and the maths was  done by me ,  except the visualization , took from youtube)

## Dataset

The model is trained on a synthetic binary classification dataset generated using `sklearn.datasets.make_blobs`: four Gaussian clusters, relabeled into two classes arranged diagonally so that **no single straight line can separate them** (a linearly non-separable dataset). This is intentional — it's the reason the model needs a hidden layer instead of a single perceptron.

```python
from sklearn.datasets import make_blobs

m = 2000
samples, labels = make_blobs(n_samples=m,
                              centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
                              cluster_std=1.1,
                              random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0

X = samples.T                 # shape (2, m)
Y = labels.reshape(1, m)       # shape (1, m)
```

## Architecture

```
Input (n_x = 2)  →  Hidden layer (n_h, tunable)  →  Output (n_y = 1)
```

- **n_x = 2** — each data point has 2 coordinates.
- **n_h** — hidden layer size, a tunable hyperparameter. Tested with `n_h = 2` (the minimum needed to separate this 4-cluster pattern) and `n_h = 5` (more capacity, same accuracy on this dataset — see Results).
- **n_y = 1** — single output neuron producing a probability via sigmoid.

## Step-by-step derivation

### 1. Forward propagation

For the full dataset (vectorized across all `m` examples):

$$Z^{[1]} = W^{[1]}X + b^{[1]}, \quad A^{[1]} = \sigma(Z^{[1]})$$

$$Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}, \quad A^{[2]} = \sigma(Z^{[2]})$$

where $\sigma(z) = \dfrac{1}{1+e^{-z}}$ is the sigmoid activation.

**Parameter shapes**, derived from matrix multiplication rules (`(a,b) @ (b,c) = (a,c)`):

| Parameter | Shape | Reasoning |
|---|---|---|
| `W1` | `(n_h, n_x)` | maps `n_x` inputs → `n_h` hidden outputs |
| `b1` | `(n_h, 1)` | one bias per hidden neuron |
| `W2` | `(n_y, n_h)` | maps `n_h` hidden inputs → `n_y` output |
| `b2` | `(n_y, 1)` | one bias per output neuron |

`W1` and `W2` are initialized with small random values (`* 0.01`) to break symmetry between neurons and keep initial activations near sigmoid's steepest, most sensitive region. `b1` and `b2` are initialized to zero, since the random weights alone are enough to make neurons distinct from each other.

### 2. Cost function

Per-example binary cross-entropy (log loss):

$$L^{(i)} = -y^{(i)}\log(a^{[2] (i)}) - (1-y^{(i)})\log(1-a^{[2] (i)})$$

Averaged over all `m` examples:

$$\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m} L^{(i)}$$

This penalizes confident wrong predictions heavily (loss → ∞ as a confident prediction approaches the wrong class) and rewards confident correct predictions with near-zero cost.

### 3. Backpropagation — the chain rule, applied layer by layer

The dependency chain from `W1` to the cost is:

$$W^{[1]} \to Z^{[1]} \to A^{[1]} \to Z^{[2]} \to A^{[2]} \to \mathcal{L}$$

By the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{\partial \mathcal{L}}{\partial A^{[2]}} \cdot \frac{\partial A^{[2]}}{\partial Z^{[2]}} \cdot \frac{\partial Z^{[2]}}{\partial A^{[1]}} \cdot \frac{\partial A^{[1]}}{\partial Z^{[1]}} \cdot \frac{\partial Z^{[1]}}{\partial W^{[1]}}$$

**Output layer gradients.** Log loss combined with a sigmoid output collapses to a clean result:

$$dZ^{[2]} = A^{[2]} - Y$$

$$\frac{\partial \mathcal{L}}{\partial W^{[2]}} = \frac{1}{m}(dZ^{[2]})(A^{[1]})^T, \qquad \frac{\partial \mathcal{L}}{\partial b^{[2]}} = \frac{1}{m}\sum dZ^{[2]}$$

**Hidden layer gradients.** The error signal is propagated backward through `W2`, then scaled by the sigmoid derivative:

$$dZ^{[1]} = \left((W^{[2]})^T dZ^{[2]}\right) \odot A^{[1]}(1-A^{[1]})$$

$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \frac{1}{m}(dZ^{[1]})X^T, \qquad \frac{\partial \mathcal{L}}{\partial b^{[1]}} = \frac{1}{m}\sum dZ^{[1]}$$

($\odot$ denotes element-wise multiplication.)

### 4. Gradient descent

Parameters are updated in the direction that reduces cost, scaled by a learning rate $\alpha$:

$$W \leftarrow W - \alpha\frac{\partial \mathcal{L}}{\partial W}, \qquad b \leftarrow b - \alpha\frac{\partial \mathcal{L}}{\partial b}$$

Repeated for `num_iterations` steps until the cost converges.

## Training loop

```
for i in range(num_iterations):
    A2, cache = forward_propagation(X, parameters)
    cost      = compute_cost(A2, Y)
    grads     = backward_propagation(parameters, cache, X, Y)
    parameters = update_parameters(parameters, grads, learning_rate)
```

## Results

| `n_h` | Behavior |
|---|---|
| 2 | Minimum viable size — successfully separates the 4-cluster diagonal pattern |
| 5 | More capacity, comparable accuracy on this dataset — extra neurons don't help further since the underlying pattern doesn't require more complexity |

## What this project deliberately does *not* use

No TensorFlow, PyTorch, or autograd — every gradient above is hand-derived and hand-implemented, to build a first-principles understanding of backpropagation before using higher-level frameworks.

## Reference

Structure and formulas referenced from DeepLearning.AI's *Math for Machine Learning* specialization. Implementation, debugging, and experimentation done independently.






