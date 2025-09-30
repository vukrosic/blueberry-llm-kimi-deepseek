# Neuron From Scratch: What is a Neuron?

Welcome to the "Neuron From Scratch" series! Here, we'll build a single artificial neuron, the fundamental building block of all neural networks.

## 1. The Goal: Understand the Concept of a Neuron

Before we write any code, let's understand the idea behind a neuron. What is it trying to do?

**A neuron is a tiny decision-making unit.** It takes in several pieces of evidence, weighs how important each piece is, and then decides how strongly to "fire" based on that evidence.

## 2. A Real-World Analogy: Should I Go for a Run?

Imagine you're deciding whether or not to go for a run. You might consider a few factors (inputs):
1.  **Is the weather good?** (Yes/No)
2.  **Do I have enough time?** (Yes/No)
3.  **Is my friend going with me?** (Yes/No)

These are your **inputs**.

Now, not all factors are equally important to you.
- The weather is very important. You hate running in the rain. Let's give this a **weight** of 5.
- Time is also important. Let's give this a **weight** of 3.
- Your friend joining is a nice bonus, but not essential. Let's give this a **weight** of 1.

These are your **weights**. Weights determine the importance of each input.

Finally, you have your own internal motivation. Maybe you're naturally a bit lazy and need a lot of convincing to go for a run. This is your **bias**. It's a single number that represents a thumb on the scale, either for or against the decision. Let's say your "laziness" bias is -4.

## 3. The Neuron's Calculation

A neuron does exactly this! It combines the evidence in two steps:

**Step 1: The Linear Step**
The neuron calculates a "score" by multiplying each input by its weight and then adding the bias.

`Score = (input_1 * weight_1) + (input_2 * weight_2) + (input_3 * weight_3) + bias`

Let's say the weather is good (input=1), you have time (input=1), but your friend isn't coming (input=0).

`Score = (1 * 5) + (1 * 3) + (0 * 1) + (-4)`
`Score = 5 + 3 + 0 - 4`
`Score = 4`

**Step 2: The Activation Step**
The neuron doesn't just output the raw score. It passes this score through an **activation function** to decide how strongly to "fire". The activation function squashes the score into a specific range (e.g., between 0 and 1).

If the score is high, the activation function will output a high value (e.g., 0.9, meaning "Yes, definitely go for a run!").
If the score is low, it will output a low value (e.g., 0.1, meaning "No, don't go.").

`Final Output = ActivationFunction(Score)`
`Final Output = ActivationFunction(4)`

We'll learn about specific activation functions in the next lesson!

## 4. Key Parts of a Neuron

- **Inputs (`x`)**: The data or evidence given to the neuron.
- **Weights (`w`)**: The importance of each input. **These are the main values that a neuron "learns"**.
- **Bias (`b`)**: An overall offset or "thumb on the scale" that helps the neuron make a decision. This is also learned.
- **Activation Function**: A function that processes the neuron's internal score to produce the final output.

---
Next, we'll look closer at the linear step of the calculation.

---

**Next Lesson**: [The Linear Step](02_the_linear_step.md)
