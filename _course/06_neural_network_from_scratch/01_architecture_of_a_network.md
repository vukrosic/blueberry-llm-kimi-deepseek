# Neural Network From Scratch: Architecture of a Network

Welcome to the "Neural Network From Scratch" series! We've built a single neuron. Now, we'll assemble them into a fully working neural network. In this lesson, we'll cover the high-level architecture.

## 1. The Goal: Understand the Layers of a Network

A single neuron is not very powerful. The power of deep learning comes from organizing these neurons into **layers**, and stacking these layers to form a **network**.

A typical feed-forward neural network has three types of layers:
1.  An **Input Layer**
2.  One or more **Hidden Layers**
3.  An **Output Layer**

<img src="https://i.imgur.com/lF34a2r.png" width="600">

Information flows from left to right: from the input layer, through the hidden layers, to the output layer. This is why it's called a **feed-forward** network.

## 2. The Input Layer

The input layer isn't really a layer of neurons. It's simply a layer that represents the raw input data.

- The number of "nodes" in the input layer is equal to the number of features in your data.
- For example, if you are predicting house prices based on `(square_footage, num_bedrooms, age)`, the input layer would have 3 nodes.
- If you are using the 28x28 MNIST images of handwritten digits, you would typically flatten the image into a vector of `28 * 28 = 784` features. The input layer would have 784 nodes.

## 3. The Hidden Layers

The hidden layers are the true "brain" of the network. They are called "hidden" because they have no direct connection to the outside world—they only receive inputs from other layers and send outputs to other layers.

- Each hidden layer is made up of a collection of neurons.
- The **width** of a layer is the number of neurons in it.
- The **depth** of a network is the number of hidden layers. A "deep" neural network has many hidden layers.

Each neuron in a hidden layer is connected to **all** the outputs from the previous layer. This is called a **fully-connected** or **dense** layer.

The hidden layers are where the network learns to find complex patterns and relationships in the data. The first hidden layer might learn to detect very simple patterns (like edges in an image), the next layer might combine those edges to detect shapes (like eyes or ears), and a later layer might combine those shapes to detect objects (like a cat).

## 4. The Output Layer

The output layer is the final layer of neurons, and its design depends entirely on the task you are trying to solve.

- The number of neurons in the output layer corresponds to the number of outputs you want.
- The activation function used in the output layer is also task-specific.

### Common Output Layer Designs:

**a) Binary Classification** (e.g., Spam or Not Spam)
- **Number of Neurons**: 1
- **Activation Function**: **Sigmoid**. The output is a single probability value between 0 and 1.

**b) Multi-Class Classification** (e.g., Cat, Dog, or Bird)
- **Number of Neurons**: `C`, where `C` is the number of classes. (e.g., 3 neurons for cat/dog/bird).
- **Activation Function**: **Softmax**. The output is a probability distribution across all `C` classes, summing to 1.

**c) Regression** (e.g., Predicting a House Price)
- **Number of Neurons**: 1
- **Activation Function**: **None**. The output is a single, unbounded number representing the predicted price.

## 5. Our Goal for this Series

We will build a simple, fully-connected neural network to solve a binary classification problem. Our architecture will be:
- **Input Layer**: 2 nodes (for 2 input features).
- **Hidden Layer 1**: 16 neurons (with ReLU activation).
- **Hidden Layer 2**: 8 neurons (with ReLU activation).
- **Output Layer**: 1 neuron (with Sigmoid activation).

---
Next, we'll move from the single `Neuron` class to a `Layer` class, which represents a collection of neurons.

---

**Next Lesson**: [Building a Layer](02_building_a_layer.md)
