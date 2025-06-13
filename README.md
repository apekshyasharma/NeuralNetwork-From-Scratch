Neural Network from Scratch – MNIST Digit Classifier
=================================================

A complete implementation of a two-layer neural network built from scratch using NumPy to classify handwritten digits from the MNIST dataset. This project demonstrates fundamental deep learning concepts without relying on high-level frameworks.

Project Overview
---------------

This neural network achieves 90.6% accuracy on the MNIST digit classification task using only basic Python libraries. The implementation includes forward propagation, backpropagation, and gradient descent optimization - all built from the ground up.

Architecture:
```
Input Layer (784) → Hidden Layer (10) → Output Layer (10)
```

- Input Layer: 784 neurons (28×28 pixel images flattened)
- Hidden Layer: 10 neurons with ReLU activation  
- Output Layer: 10 neurons with Softmax activation (digits 0-9)

Features
--------

- Built from scratch - No TensorFlow/PyTorch dependencies
- Complete implementation - Forward prop, backprop, gradient descent
- Data preprocessing - Normalization and train/dev split
- Visualization tools - Display predictions with actual images
- High accuracy - Achieves 90%+ accuracy on test data
- Educational - Well-documented code perfect for learning

Requirements
------------

```
numpy>=1.19.0
pandas>=1.3.0
matplotlib>=3.3.0
```

Quick Start
-----------

1. Install Dependencies
```bash
pip install numpy pandas matplotlib
```

2. Download MNIST Data
Download the MNIST dataset files:
- train.csv - Training data (42,000 samples)
- test.csv - Test data (28,000 samples)

3. Run the Code
```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and run the neural network
# (Copy the complete code from the notebook)
```

4. Train the Model
```python
# Train for 500 iterations with learning rate 0.1
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 500, 0.1)
```

Core Components
---------------

Data Preprocessing:
- Normalization: Pixel values scaled to [0,1] range
- Shuffling: Random data ordering to prevent bias
- Train/Dev Split: 41,000 training + 1,000 development samples

Neural Network Functions:

| Function | Purpose |
|----------|---------|
| parameters() | Initialize weights and biases |
| ReLU() | Rectified Linear Unit activation |
| softmax() | Convert outputs to probabilities |
| forward_propagation() | Compute network output |
| back_propagation() | Calculate gradients |
| gradient_descent() | Training loop with parameter updates |

Key Algorithms:
- Forward Propagation: Input → Hidden → Output
- Backpropagation: Calculate gradients using chain rule
- Gradient Descent: Optimize weights using computed gradients

Performance
-----------

Training Progress:
```
Iteration 0   | Accuracy: 10.87%
Iteration 100 | Accuracy: 85.18%
Iteration 200 | Accuracy: 88.19%
Iteration 300 | Accuracy: 89.53%
Iteration 400 | Accuracy: 90.27%
Iteration 490 | Accuracy: 90.75%
```

Final Results:
- Training Accuracy: 90.75%
- Development Accuracy: 90.6%
- Training Time: ~500 iterations

Visualization
-------------

The project includes visualization tools to:
- Display individual digit predictions
- Compare predicted vs actual labels
- Show misclassified examples for analysis

```python
# Test prediction on specific image
test_predictions(index=10, W1, b1, W2, b2)
# Output: Shows image with prediction and true label
```

Educational Value
----------------

This implementation is perfect for understanding:
- Neural Network Fundamentals: How neurons connect and process information
- Backpropagation Algorithm: How networks learn from mistakes
- Gradient Descent: How optimization improves performance
- Matrix Operations: Efficient computation using NumPy
- Classification Tasks: Multi-class prediction problems

Project Structure
----------------

```
mnist-neural-network/
├── README.md
├── requirements.txt
├── neural_network.py        # Main implementation
├── data/
│   ├── train.csv           # Training dataset
│   └── test.csv            # Test dataset
└── notebooks/
    └── mnist_classifier.ipynb  # Jupyter notebook version
```

Code Highlights
---------------

Network Initialization:
```python
def parameters():
    W1 = np.random.randn(10, 784) * np.sqrt(2. / 784)  # He initialization
    b1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2. / 10)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2
```

Forward Propagation:
```python
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1      # Linear transformation
    A1 = ReLU(Z1)            # Non-linear activation
    Z2 = W2.dot(A1) + b2     # Second layer
    A2 = softmax(Z2)         # Output probabilities
    return Z1, A1, Z2, A2
```

Contributing
------------

Contributions are welcome! Areas for improvement:
- Add more activation functions (tanh, sigmoid)
- Implement different optimizers (Adam, RMSprop)
- Add regularization techniques (dropout, L2)
- Create more visualization tools
- Optimize performance with vectorization

Learning Resources
-----------------

- Neural Networks and Deep Learning: http://neuralnetworksanddeeplearning.com/
- CS231n: Convolutional Neural Networks: http://cs231n.github.io/
- Deep Learning Specialization: https://www.coursera.org/specializations/deep-learning

License
-------

This project is open source and available under the MIT License.

Dataset Source
--------------

This project uses the MNIST digit recognition dataset from Kaggle:
- Dataset: https://www.kaggle.com/competitions/digit-recognizer
- Files: train.csv (42,000 samples) and test.csv (28,000 samples)

Tutorial Credit
---------------

This implementation is based on the excellent tutorial by Samson Zhang:
- YouTube Tutorial: "Neural Networks from Scratch" 
- Video Link: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1s
- Channel: Samson Zhang

The tutorial provides a clear, step-by-step explanation of building neural networks from scratch without using high-level deep learning frameworks.

Acknowledgments
---------------

- MNIST Dataset: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
- Tutorial Creator: Samson Zhang for the comprehensive YouTube explanation
- Dataset Platform: Kaggle for hosting the MNIST digit recognizer competition
- Community: Thanks to all contributors and learners

---

Star this repo if you found it helpful!

Built with ❤️ for the machine learning community
