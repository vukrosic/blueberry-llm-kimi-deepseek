# Advanced Python Features

## Learning Objectives
- Master decorators and their applications
- Understand generators and iterators
- Learn about context managers
- Practice with advanced Python techniques

## Decorators

### What are Decorators?
Decorators are functions that modify the behavior of other functions without changing their code. Think of them as wrappers that add functionality.

### Basic Decorator
```python
def my_decorator(func):
    def wrapper():
        print("Something is happening before the function is called.")
        func()
        print("Something is happening after the function is called.")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

# Using the decorator
say_hello()
# Output:
# Something is happening before the function is called.
# Hello!
# Something is happening after the function is called.
```

### Decorator with Arguments
```python
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Output:
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!
```

### Built-in Decorators
```python
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
    
    @staticmethod
    def is_valid_radius(radius):
        return radius > 0
    
    @classmethod
    def from_diameter(cls, diameter):
        return cls(diameter / 2)

# Using the decorators
circle = Circle(5)
print(circle.area)  # 78.53975

circle.radius = 10
print(circle.area)  # 314.159

print(Circle.is_valid_radius(5))  # True
print(Circle.from_diameter(20).radius)  # 10.0
```

## Generators and Iterators

### What are Generators?
Generators are functions that return an iterator. They use `yield` instead of `return` and can pause and resume execution.

### Basic Generator
```python
def count_up_to(max_count):
    count = 1
    while count <= max_count:
        yield count
        count += 1

# Using the generator
counter = count_up_to(5)
for number in counter:
    print(number)
# Output: 1, 2, 3, 4, 5

# Or using next()
counter = count_up_to(3)
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3
```

### Generator Expressions
```python
# List comprehension (creates list in memory)
squares_list = [x**2 for x in range(10)]
print(squares_list)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Generator expression (lazy evaluation)
squares_gen = (x**2 for x in range(10))
print(squares_gen)  # <generator object <genexpr> at 0x...>

# Convert to list when needed
print(list(squares_gen))  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

### Practical Generator Example
```python
def fibonacci_generator(n):
    a, b = 0, 1
    count = 0
    while count < n:
        yield a
        a, b = b, a + b
        count += 1

# Using the generator
fib = fibonacci_generator(10)
for number in fib:
    print(number, end=" ")
# Output: 0 1 1 2 3 5 8 13 21 34
```

## Context Managers

### What are Context Managers?
Context managers ensure that resources are properly managed (opened and closed) even if an error occurs. They use the `with` statement.

### Basic Context Manager
```python
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# Using the context manager
with FileManager("data.txt", "w") as file:
    file.write("Hello, World!")
# File is automatically closed here
```

### Using `contextlib`
```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# Using the context manager
with file_manager("data.txt", "r") as file:
    content = file.read()
    print(content)
```

### Built-in Context Managers
```python
import os
from contextlib import suppress

# Suppressing exceptions
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")

# Working with files
with open("data.txt", "w") as file:
    file.write("Hello, World!")

# Working with locks (threading)
import threading
lock = threading.Lock()

with lock:
    # Critical section code
    print("This code is thread-safe")
```

## Lambda Functions

### Basic Lambda Functions
```python
# Regular function
def square(x):
    return x ** 2

# Lambda function
square_lambda = lambda x: x ** 2

print(square(5))        # 25
print(square_lambda(5)) # 25

# Lambda with multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7
```

### Using Lambda with Built-in Functions
```python
# With map()
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With filter()
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(even_numbers)  # [2, 4]

# With sorted()
students = [("Alice", 85), ("Bob", 90), ("Charlie", 78)]
sorted_by_grade = sorted(students, key=lambda x: x[1], reverse=True)
print(sorted_by_grade)  # [("Bob", 90), ("Alice", 85), ("Charlie", 78)]
```

## List Comprehensions (Advanced)

### Nested List Comprehensions
```python
# Flatten a nested list
nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested_list for item in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create a multiplication table
table = [[i * j for j in range(1, 6)] for i in range(1, 6)]
for row in table:
    print(row)
```

### Dictionary and Set Comprehensions
```python
# Dictionary comprehension
squares_dict = {x: x**2 for x in range(1, 6)}
print(squares_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Set comprehension
unique_squares = {x**2 for x in range(-5, 6)}
print(unique_squares)  # {0, 1, 4, 9, 16, 25}

# Conditional comprehensions
even_squares = {x: x**2 for x in range(1, 11) if x % 2 == 0}
print(even_squares)  # {2: 4, 4: 16, 6: 36, 8: 64, 10: 100}
```

## Practice Exercises

### Exercise 1: Decorator for Timing Functions
Create a decorator that:
- Measures how long a function takes to execute
- Prints the execution time
- Works with functions that have arguments
- Can be applied to multiple functions

### Exercise 2: Generator for Prime Numbers
Write a generator that:
- Yields prime numbers up to a given limit
- Uses the Sieve of Eratosthenes algorithm
- Can be used to find the first N prime numbers
- Is memory efficient for large ranges

### Exercise 3: Context Manager for Database Connections
Create a context manager that:
- Simulates database connection management
- Handles connection errors gracefully
- Logs connection operations
- Ensures connections are always closed

### Exercise 4: Advanced Data Processing
Build a data processing system that:
- Uses generators to process large datasets
- Applies multiple transformations using decorators
- Handles errors with context managers
- Uses comprehensions for data manipulation

## Common Advanced Patterns

### Memoization with Decorators
```python
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args in cache:
            return cache[args]
        result = func(*args)
        cache[args] = result
        return result
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Much faster for large numbers
print(fibonacci(100))  # 354224848179261915075
```

### Chaining Decorators
```python
def bold(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"<b>{result}</b>"
    return wrapper

def italic(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return f"<i>{result}</i>"
    return wrapper

@bold
@italic
def say_hello(name):
    return f"Hello, {name}!"

print(say_hello("Alice"))  # <b><i>Hello, Alice!</i></b>
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning advanced Python features - decorators, generators, context managers, and lambda functions. I understand basic Python and OOP but I'm struggling with:

1. How decorators work and when to use them
2. The difference between generators and regular functions
3. When to use context managers vs try-finally blocks
4. Practical applications of lambda functions
5. Advanced list/dict/set comprehensions
6. How to combine these features effectively

Please:
- Explain each concept with clear, practical examples
- Show me real-world use cases for each feature
- Help me understand when to choose one approach over another
- Walk me through building complex systems using these features
- Give me exercises that combine multiple advanced concepts
- Explain performance implications and best practices

I want to write Pythonic, efficient code using these advanced features. Please provide hands-on examples and help me think like an advanced Python developer."

## Key Takeaways
- Decorators add functionality without modifying original code
- Generators are memory-efficient for large datasets
- Context managers ensure proper resource management
- Lambda functions are useful for simple operations
- Advanced comprehensions can replace complex loops
- Combine features for powerful, elegant solutions

## Next Steps
Master these advanced features and you'll be ready for:
- Building production-ready applications
- Working with frameworks and libraries
- Performance optimization
- Preparing for AI/ML development
- Contributing to open source projects
