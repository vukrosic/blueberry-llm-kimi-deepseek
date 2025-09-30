
# Lesson 5: Calculating Gradients for a Single Neuron

Now that we understand the chain rule, let's apply it to a single neuron. Our goal is to figure out how to adjust the neuron's weights and biases to reduce the loss.

## Recap: A Single Neuron

A neuron performs two steps:

1.  **Linear Step:** `z = w * x + b` (for a single input `x` and weight `w`)
2.  **Activation Step:** `a = f(z)` (where `f` is an activation function like ReLU or Sigmoid)

The output `a` of our neuron is then used to calculate the total loss of the network, which we'll call `L`.

We want to calculate the gradients of the loss with respect to the neuron's parameters, `w` and `b`. These are:

- `dL/dw`: How the loss changes when we change the weight `w`.
- `dL/db`: How the loss changes when we change the bias `b`.

## Using the Chain Rule

To find `dL/dw`, we can chain together the derivatives:

`L` depends on `a`.
`a` depends on `z`.
`z` depends on `w`.

So, the chain rule gives us:

```
dL/dw = dL/da * da/dz * dz/dw
```

Let's break this down:

1.  `dz/dw`: The derivative of the linear step `z = w * x + b` with respect to `w`. This is simply `x`.

2.  `da/dz`: The derivative of the activation function `a = f(z)` with respect to `z`. This depends on the activation function we choose. For example, the derivative of the sigmoid function is `sigmoid(z) * (1 - sigmoid(z))`.

3.  `dL/da`: The derivative of the loss with respect to the neuron's output `a`. This term represents how much the final loss is affected by this neuron's output. When we backpropagate, this is the gradient that comes from the *next* layer.

So, the full equation for the gradient of the weight is:

```
dL/dw = (dL/da) * f'(z) * x
```

Where `f'(z)` is the derivative of the activation function.

## Gradient for the Bias

We can do the same for the bias `b`:

```
dL/db = dL/da * da/dz * dz/db
```

The only difference is `dz/db`. The derivative of `z = w * x + b` with respect to `b` is `1`.

So, the equation for the gradient of the bias is:

```
dL/db = (dL/da) * f'(z)
```

## What this Means

We now have a way to calculate the gradients for a single neuron! These gradients tell us the direction to adjust our weights and biases to decrease the loss.

- If `dL/dw` is positive, it means increasing `w` increases the loss. So we should decrease `w`.
- If `dL/dw` is negative, increasing `w` decreases the loss. So we should increase `w`.

This is the core of gradient-based learning. In the next lesson, we'll see how this works in a full network with multiple layers.

---

**Next Lesson**: [Backpropagation in Action](06_backpropagation_in_action.md)
