# File Handling and Modules in Python

## Learning Objectives
- Learn to read from and write to files
- Understand Python modules and packages
- Master importing and using external libraries
- Practice with real file operations

## File Handling - Working with Files

### Reading Files
```python
# Reading entire file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

# Reading line by line
with open("data.txt", "r") as file:
    for line in file:
        print(line.strip())  # strip() removes newline characters

# Reading all lines into a list
with open("data.txt", "r") as file:
    lines = file.readlines()
    print(lines)
```

### Writing Files
```python
# Writing to a file
data = "Hello, World!\nThis is a new line."

with open("output.txt", "w") as file:
    file.write(data)

# Appending to a file
with open("output.txt", "a") as file:
    file.write("\nThis line was appended.")

# Writing multiple lines
lines = ["Line 1\n", "Line 2\n", "Line 3\n"]

with open("output.txt", "w") as file:
    file.writelines(lines)
```

### File Modes
```python
# Different file modes
with open("file.txt", "r") as file:    # Read mode (default)
    pass

with open("file.txt", "w") as file:    # Write mode (overwrites)
    pass

with open("file.txt", "a") as file:    # Append mode
    pass

with open("file.txt", "r+") as file:   # Read and write
    pass

with open("file.txt", "x") as file:    # Exclusive creation
    pass
```

### Working with CSV Files
```python
import csv

# Reading CSV files
with open("students.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

# Writing CSV files
data = [
    ["Name", "Age", "Grade"],
    ["Alice", "20", "A"],
    ["Bob", "19", "B"],
    ["Charlie", "21", "A"]
]

with open("output.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(data)
```

### Working with JSON Files
```python
import json

# Reading JSON files
with open("data.json", "r") as file:
    data = json.load(file)
    print(data)

# Writing JSON files
student_data = {
    "name": "Alice",
    "age": 20,
    "grades": [85, 90, 78],
    "is_student": True
}

with open("student.json", "w") as file:
    json.dump(student_data, file, indent=2)
```

## Modules and Packages

### What are Modules?
Modules are Python files that contain functions, classes, and variables that you can use in other programs.

### Creating Your Own Module
```python
# math_utils.py
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def power(base, exponent):
    return base ** exponent

PI = 3.14159
```

### Importing Modules
```python
# Import entire module
import math_utils

result = math_utils.add(5, 3)
print(result)

# Import specific functions
from math_utils import add, multiply

result = add(5, 3)
print(result)

# Import with alias
import math_utils as mu

result = mu.multiply(4, 6)
print(result)

# Import everything (not recommended)
from math_utils import *
```

### Built-in Modules
```python
import random
import datetime
import os
import sys

# Random numbers
number = random.randint(1, 10)
choice = random.choice(["apple", "banana", "orange"])

# Date and time
now = datetime.datetime.now()
today = datetime.date.today()

# Operating system
current_dir = os.getcwd()
files = os.listdir(".")

# System information
python_version = sys.version
```

### Third-party Packages
```python
# Installing packages (run in terminal)
# pip install requests

import requests

# Making HTTP requests
response = requests.get("https://api.github.com/users/octocat")
data = response.json()
print(data["name"])
```

## Practice Exercises

### Exercise 1: Text File Processor
Create a program that:
- Reads a text file
- Counts words, lines, and characters
- Finds the most common word
- Saves statistics to a new file

### Exercise 2: Student Database
Build a simple database system:
- Store student information in a JSON file
- Add, update, delete, and search students
- Export data to CSV format
- Handle file errors gracefully

### Exercise 3: Log File Analyzer
Write a program that:
- Reads a log file (simulated)
- Analyzes error patterns
- Generates a summary report
- Saves results to multiple formats

### Exercise 4: Configuration Manager
Create a configuration system:
- Read settings from a JSON file
- Allow runtime configuration changes
- Save updated settings
- Provide default values for missing settings

## Common Patterns

### File Error Handling
```python
try:
    with open("data.txt", "r") as file:
        content = file.read()
except FileNotFoundError:
    print("File not found!")
except PermissionError:
    print("Permission denied!")
except Exception as e:
    print(f"An error occurred: {e}")
```

### Path Operations
```python
import os
from pathlib import Path

# Using os module
current_dir = os.getcwd()
file_path = os.path.join(current_dir, "data.txt")
exists = os.path.exists(file_path)

# Using pathlib (more modern)
path = Path("data.txt")
exists = path.exists()
parent = path.parent
name = path.name
```

### Working with Directories
```python
import os

# Create directory
os.makedirs("new_folder", exist_ok=True)

# List files
files = os.listdir(".")
for file in files:
    if file.endswith(".py"):
        print(f"Python file: {file}")

# Walk through directory tree
for root, dirs, files in os.walk("."):
    for file in files:
        print(os.path.join(root, file))
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning about file handling and modules in Python. I understand basic Python but I'm struggling with:

1. How to properly read and write different file types (text, CSV, JSON)
2. When to use different file modes (r, w, a, r+)
3. How to create and import my own modules
4. Working with built-in modules like os, sys, datetime
5. Installing and using third-party packages
6. Best practices for file operations and error handling

Please:
- Explain file operations with practical examples
- Show me how to handle different file formats
- Walk me through creating reusable modules
- Help me understand the Python package ecosystem
- Give me exercises that combine file handling with other concepts
- Explain common errors and how to avoid them

I want to build programs that can persist data and work with external files. Please provide hands-on examples and real-world scenarios."

## Key Takeaways
- Always use `with` statements for file operations
- Choose the right file mode for your needs
- Handle file errors gracefully
- Modules make code reusable and organized
- Python has a rich ecosystem of packages
- Practice with real file operations

## Next Steps
Master file handling and modules and you'll be ready for:
- Object-oriented programming
- Working with databases
- Building web applications
- Data analysis and visualization
- Creating your own Python packages
