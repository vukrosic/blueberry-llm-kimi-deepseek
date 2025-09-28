## **No need to be able to code all of these lessons from memory, but be able to read and understand what each does.**

# Python Basics - Variables, Data Types, and Functions

## Learning Objectives
- Understand what Python is and why it's used
- Learn about variables and data types
- Master basic functions
- Practice with hands-on examples

## What is Python?
Python is a high-level programming language that's:
- **Easy to read** - looks like English
- **Versatile** - used for web development, data science, AI, automation
- **Beginner-friendly** - great first programming language
- **Powerful** - used by Google, Netflix, Instagram, and more

## Variables and Data Types

### Variables
Variables are like labeled boxes that store information.

You can also run this on [Google Colab](https://www.youtube.com/watch?v=RLYoEyIHL6A)

```python
# Creating variables
name = "Alice"
age = 25
height = 5.6
is_student = True
```

### Data Types
Python has several built-in data types:

1. **Strings** - Text data (in quotes)
2. **Integers** - Whole numbers
3. **Floats** - Decimal numbers
4. **Booleans** - True or False
5. **Lists** - Ordered collections
6. **Dictionaries** - Key-value pairs

```python
# Examples of different data types
name = "Python"           # String
version = 3.9             # Integer
price = 0.0               # Float
is_free = True            # Boolean
languages = ["Python", "Java", "C++"]  # List
person = {"name": "Alice", "age": 25}  # Dictionary
```

## Functions

Functions are reusable blocks of code that perform specific tasks.

### Creating Functions
```python
def greet(name):
    return f"Hello, {name}!"

def add_numbers(a, b):
    return a + b

def calculate_area(length, width):
    area = length * width
    return area
```

### Using Functions
```python
# Call the functions
message = greet("Alice")
print(message)  # Output: Hello, Alice!

result = add_numbers(5, 3)
print(result)   # Output: 8

room_area = calculate_area(10, 12)
print(f"Room area: {room_area} square feet")
```

## Practice Exercises

### Exercise 1: Variable Practice
Create variables for:
- Your name (string)
- Your age (integer)
- Your favorite number (float)
- Whether you like programming (boolean)

### Exercise 2: Function Practice
Write a function called `calculate_circle_area` that takes a radius and returns the area of a circle.
(Hint: area = π × radius², use 3.14159 for π)

### Exercise 3: String Functions
Write a function that takes a name and returns:
- The name in uppercase
- The name in lowercase
- The length of the name

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning Python basics and need help with variables, data types, and functions. Please:

1. Explain what variables are using a simple analogy (like labeled boxes)
2. Show me examples of each data type (string, integer, float, boolean, list, dictionary)
3. Help me understand what functions are and why we use them
4. Walk me through creating my first function step by step
5. Give me practice exercises with solutions
6. Explain any errors I make and how to fix them
7. Use simple, beginner-friendly language
8. Let me practice each concept before moving to the next

I want to understand the WHY behind each concept, not just the HOW. Please be patient and provide clear examples."

## Key Takeaways
- Variables store data with meaningful names
- Python automatically determines data types
- Functions make code reusable and organized
- Practice is essential for learning programming
- Don't be afraid to make mistakes - they're part of learning!

## Next Steps
Once you master these basics, you'll be ready for:
- Control flow (if statements, loops)
- Lists and data structures
- Object-oriented programming
- Building your first AI/ML models
