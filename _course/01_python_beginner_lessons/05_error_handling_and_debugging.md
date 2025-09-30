# Error Handling and Debugging in Python

## Learning Objectives
- Understand different types of Python errors
- Master try-except blocks for error handling
- Learn debugging techniques and tools
- Practice with real error scenarios

## Types of Python Errors

### Syntax Errors
```python
# Missing colon
if x > 5
    print("Hello")

# Missing parentheses
print "Hello World"

# Incorrect indentation
def hello():
print("Hello")
```

### Runtime Errors (Exceptions)
```python
# ZeroDivisionError
result = 10 / 0

# TypeError
number = 5 + "hello"

# ValueError
number = int("hello")

# IndexError
my_list = [1, 2, 3]
print(my_list[10])

# KeyError
my_dict = {"name": "Alice"}
print(my_dict["age"])

# FileNotFoundError
with open("nonexistent.txt", "r") as file:
    content = file.read()
```

## Try-Except Blocks

### Basic Error Handling
```python
try:
    number = int(input("Enter a number: "))
    result = 10 / number
    print(f"Result: {result}")
except ValueError:
    print("Please enter a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

### Multiple Exception Handling
```python
try:
    # Risky code here
    data = {"name": "Alice", "age": 25}
    print(data["name"])
    print(data["city"])  # This will cause KeyError
except KeyError as e:
    print(f"Key not found: {e}")
except TypeError as e:
    print(f"Type error: {e}")
except Exception as e:
    print(f"General error: {e}")
```

### Else and Finally Clauses
```python
try:
    number = int(input("Enter a number: "))
    result = 10 / number
except ValueError:
    print("Invalid input!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
else:
    print(f"Success! Result: {result}")
finally:
    print("This always runs, regardless of errors")
```

## Common Error Handling Patterns

### Input Validation
```python
def get_positive_number():
    while True:
        try:
            number = float(input("Enter a positive number: "))
            if number > 0:
                return number
            else:
                print("Number must be positive!")
        except ValueError:
            print("Please enter a valid number!")

# Usage
age = get_positive_number()
print(f"You entered: {age}")
```

### File Operations
```python
def read_file_safely(filename):
    try:
        with open(filename, "r") as file:
            return file.read()
    except FileNotFoundError:
        print(f"File '{filename}' not found!")
        return None
    except PermissionError:
        print(f"Permission denied to read '{filename}'!")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Usage
content = read_file_safely("data.txt")
if content:
    print("File read successfully!")
```

### Dictionary Operations
```python
def safe_dict_access(data, key, default=None):
    try:
        return data[key]
    except KeyError:
        print(f"Key '{key}' not found in dictionary")
        return default
    except TypeError:
        print("Invalid data type for dictionary access")
        return default

# Usage
person = {"name": "Alice", "age": 25}
name = safe_dict_access(person, "name", "Unknown")
city = safe_dict_access(person, "city", "Not specified")
```

## Debugging Techniques

### Print Debugging
```python
def calculate_average(numbers):
    print(f"Input: {numbers}")  # Debug print
    
    if not numbers:
        print("Empty list!")  # Debug print
        return 0
    
    total = sum(numbers)
    print(f"Total: {total}")  # Debug print
    
    average = total / len(numbers)
    print(f"Average: {average}")  # Debug print
    
    return average

# Usage
scores = [85, 90, 78, 92]
result = calculate_average(scores)
```

### Using Assertions
```python
def divide_numbers(a, b):
    assert b != 0, "Cannot divide by zero!"
    assert isinstance(a, (int, float)), "First argument must be a number"
    assert isinstance(b, (int, float)), "Second argument must be a number"
    
    return a / b

# Usage
try:
    result = divide_numbers(10, 2)
    print(f"Result: {result}")
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

### Logging for Debugging
```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def process_data(data):
    logging.debug(f"Processing data: {data}")
    
    if not data:
        logging.warning("Empty data received")
        return []
    
    try:
        result = [x * 2 for x in data]
        logging.info(f"Successfully processed {len(result)} items")
        return result
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return []

# Usage
numbers = [1, 2, 3, 4, 5]
processed = process_data(numbers)
```

## Practice Exercises

### Exercise 1: Calculator with Error Handling
Create a calculator that:
- Handles division by zero
- Validates input types
- Provides helpful error messages
- Continues running until user quits

### Exercise 2: File Processor
Build a file processor that:
- Handles missing files gracefully
- Validates file formats
- Logs all operations
- Recovers from partial failures

### Exercise 3: Data Validator
Write a data validation system:
- Validates email addresses
- Checks age ranges
- Ensures required fields are present
- Provides detailed error reports

### Exercise 4: Network Request Handler
Create a network request handler:
- Handles connection timeouts
- Retries failed requests
- Logs all network operations
- Provides fallback options

## Common Debugging Tools

### Python Debugger (pdb)
```python
import pdb

def complex_function(data):
    pdb.set_trace()  # Set breakpoint here
    result = []
    for item in data:
        processed = item * 2
        result.append(processed)
    return result

# Usage
data = [1, 2, 3, 4, 5]
result = complex_function(data)
```

### IDE Debugging
- Set breakpoints in your IDE
- Step through code line by line
- Inspect variable values
- Watch expressions change

### Error Tracing
```python
import traceback

def risky_operation():
    try:
        # Some risky code
        result = 10 / 0
    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()
        return None

risky_operation()
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning about error handling and debugging in Python. I understand basic Python but I'm struggling with:

1. Different types of errors and when they occur
2. How to use try-except blocks effectively
3. When to use else and finally clauses
4. Best practices for error handling
5. Debugging techniques and tools
6. How to write robust, error-resistant code

Please:
- Explain each error type with clear examples
- Show me different error handling patterns
- Help me understand when to catch specific vs general exceptions
- Walk me through debugging techniques step by step
- Give me exercises that involve handling real-world errors
- Explain how to write code that fails gracefully

I want to write programs that handle errors gracefully and are easy to debug. Please provide practical examples and common scenarios."

## Key Takeaways
- Errors are normal - handle them gracefully
- Use specific exception types when possible
- Always provide helpful error messages
- Debug systematically, not randomly
- Test your error handling code
- Log important operations for debugging

## Next Steps
Master error handling and debugging and you'll be ready for:
- Object-oriented programming
- Building robust applications
- Working with external APIs
- Creating production-ready code
- Advanced debugging techniques

---

**Next Lesson**: [Object-Oriented Programming](06_object_oriented_programming.md)
