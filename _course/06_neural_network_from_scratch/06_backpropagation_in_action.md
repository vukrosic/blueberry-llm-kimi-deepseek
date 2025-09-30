
# Lesson 6: Backpropagation in Action

We've seen how to calculate gradients for a single neuron. Now, let's see how it works in a full (but small) neural network. This is where the "backpropagation" part really comes to life.

## A Simple 2-Layer Network

Imagine a network with one hidden layer and one output layer:

1.  **Input** `x`
2.  **Hidden Layer:**
    -   `z1 = W1 * x + b1`
    -   `a1 = f(z1)`
3.  **Output Layer:**
    -   `z2 = W2 * a1 + b2`
    -   `a2 = f(z2)` (our final prediction)
4.  **Loss:** `L` is calculated based on `a2` and the true label `y`.

Our goal is to calculate the gradients for all the weights and biases: `W1`, `b1`, `W2`, and `b2`.

## The Backward Pass

We start from the end of the network and move backward. Hence, "backpropagation."

### Step 1: Gradients for the Output Layer (W2, b2)

This is just like the single neuron case we saw in the last lesson.

-   `dL/dW2 = (dL/da2) * f'(z2) * a1`
-   `dL/db2 = (dL/da2) * f'(z2)`

Here, `dL/da2` is the derivative of the loss function with respect to the final prediction. For example, if we use Mean Squared Error loss, `L = (a2 - y)^2`, then `dL/da2 = 2 * (a2 - y)`.

### Step 2: Gradients for the Hidden Layer (W1, b1)

Now, we need to calculate `dL/dW1` and `dL/db1`. Let's use the chain rule again:

-   `dL/dW1 = (dL/da1) * f'(z1) * x`
-   `dL/db1 = (dL/da1) * f'(z1)`

But what is `dL/da1`? This is the crucial step. `a1` affects the loss *through* `z2`. So we need to apply the chain rule again:

`dL/da1 = dL/dz2 * dz2/da1`

Let's break this down:

-   `dL/dz2`: We already know this from the output layer calculation! It's `dL/da2 * f'(z2)`.
-   `dz2/da1`: The derivative of `z2 = W2 * a1 + b2` with respect to `a1`. This is just `W2`.

So, `dL/da1 = (dL/da2 * f'(z2)) * W2`. This is the gradient from the output layer, propagated back to the hidden layer.

Now we can substitute this back into our equations for `W1` and `b1`:

-   `dL/dW1 = ((dL/da2 * f'(z2)) * W2) * f'(z1) * x`
-   `dL/db1 = ((dL/da2 * f'(z2)) * W2) * f'(z1)`

## The Flow of Gradients

As you can see, the gradient calculation starts at the loss function and flows backward through the network.

1.  Calculate the gradient of the loss with respect to the output of the last layer.
2.  Use this to calculate the gradients for the weights and biases of the last layer.
3.  Propagate the gradient back to the previous layer.
4.  Use this propagated gradient to calculate the gradients for the weights and biases of that layer.
5.  Repeat until you reach the first layer.

This is the backpropagation algorithm! In the next and final lesson of this section, we'll see how to implement this in Python.
