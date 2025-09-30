# Lists and Data Structures in Python

## Learning Objectives
- Master Python lists and their methods
- Understand dictionaries and key-value pairs
- Learn about tuples and sets
- Practice with real-world data manipulation

## What are Data Structures?
Data structures are ways to organize and store data in your program. Think of them as different types of containers, each with their own purpose and advantages.

## Lists - Ordered Collections

### Creating Lists
```python
# Different ways to create lists
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]
mixed = ["hello", 42, 3.14, True]
empty_list = []

# Using list() function
colors = list(["red", "green", "blue"])
```

### Accessing List Elements
```python
fruits = ["apple", "banana", "orange"]

# Access by index (starts at 0)
first_fruit = fruits[0]    # "apple"
second_fruit = fruits[1]   # "banana"
last_fruit = fruits[-1]    # "orange" (negative indexing)

# Slicing lists
first_two = fruits[0:2]    # ["apple", "banana"]
last_two = fruits[-2:]     # ["banana", "orange"]
```

### List Methods
```python
fruits = ["apple", "banana"]

# Adding elements
fruits.append("orange")           # Add to end
fruits.insert(1, "grape")         # Insert at specific position
fruits.extend(["kiwi", "mango"])  # Add multiple elements

# Removing elements
fruits.remove("banana")           # Remove specific value
popped = fruits.pop()             # Remove and return last element
fruits.pop(0)                     # Remove element at index

# Other useful methods
fruits.sort()                     # Sort alphabetically
fruits.reverse()                  # Reverse the list
count = fruits.count("apple")     # Count occurrences
index = fruits.index("orange")    # Find index of element
```

### List Comprehensions
```python
# Traditional way
squares = []
for i in range(5):
    squares.append(i ** 2)

# List comprehension (more Pythonic)
squares = [i ** 2 for i in range(5)]

# With conditions
even_squares = [i ** 2 for i in range(10) if i % 2 == 0]
```

## Dictionaries - Key-Value Pairs

### Creating Dictionaries
```python
# Different ways to create dictionaries
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Using dict() function
scores = dict(math=95, science=87, english=92)

# Empty dictionary
empty_dict = {}
```

### Accessing Dictionary Values
```python
person = {"name": "Alice", "age": 25}

# Access by key
name = person["name"]        # "Alice"
age = person["age"]          # 25

# Safe access with get()
city = person.get("city", "Unknown")  # Returns "Unknown" if key doesn't exist

# Check if key exists
if "age" in person:
    print("Age is available")
```

### Dictionary Methods
```python
person = {"name": "Alice", "age": 25}

# Adding/updating values
person["city"] = "New York"        # Add new key-value pair
person.update({"job": "Engineer", "age": 26})  # Update multiple pairs

# Removing values
del person["age"]                  # Remove key-value pair
removed = person.pop("city")       # Remove and return value
person.clear()                     # Remove all pairs

# Getting information
keys = person.keys()               # Get all keys
values = person.values()           # Get all values
items = person.items()             # Get key-value pairs
```

## Tuples - Immutable Sequences

### Creating and Using Tuples
```python
# Creating tuples
coordinates = (10, 20)
colors = ("red", "green", "blue")
single_item = (42,)  # Note the comma for single item

# Accessing elements (same as lists)
x = coordinates[0]   # 10
y = coordinates[1]   # 20

# Tuples are immutable (can't be changed)
# coordinates[0] = 15  # This would cause an error

# Useful for multiple return values
def get_name_and_age():
    return "Alice", 25

name, age = get_name_and_age()  # Unpacking
```

## Sets - Unique Collections

### Creating and Using Sets
```python
# Creating sets
fruits = {"apple", "banana", "orange"}
numbers = set([1, 2, 3, 4, 5])
empty_set = set()

# Sets automatically remove duplicates
duplicates = {1, 2, 2, 3, 3, 3}  # Results in {1, 2, 3}

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}

union = set1 | set2           # {1, 2, 3, 4, 5, 6}
intersection = set1 & set2    # {3, 4}
difference = set1 - set2      # {1, 2}
```

## Practice Exercises

### Exercise 1: Student Grade Tracker
Create a program that:
- Stores student names and their grades in a dictionary
- Allows adding new students and grades
- Calculates average grade for each student
- Finds the student with the highest average

### Exercise 2: Shopping Cart
Build a shopping cart system:
- Use a list to store items
- Use a dictionary to track quantities
- Add/remove items
- Calculate total cost
- Apply discounts based on total amount

### Exercise 3: Word Frequency Counter
Write a program that:
- Takes a sentence as input
- Counts how many times each word appears
- Stores results in a dictionary
- Displays the most common words

### Exercise 4: Contact Book
Create a contact management system:
- Store contacts as dictionaries in a list
- Each contact has name, phone, email
- Add, search, update, and delete contacts
- Save contacts to a file (bonus)

## Common Patterns

### Nested Data Structures
```python
# List of dictionaries
students = [
    {"name": "Alice", "grades": [85, 90, 78]},
    {"name": "Bob", "grades": [92, 88, 95]},
    {"name": "Charlie", "grades": [76, 82, 80]}
]

# Dictionary with lists
classroom = {
    "math": ["Alice", "Bob", "Charlie"],
    "science": ["Alice", "Charlie"],
    "english": ["Bob", "Charlie"]
}
```

### Data Processing
```python
# Filter and transform data
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers and square them
even_squares = [x**2 for x in numbers if x % 2 == 0]

# Using filter and map
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
squared = list(map(lambda x: x**2, even_numbers))
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning about Python data structures - lists, dictionaries, tuples, and sets. I understand basic Python but I'm confused about:

1. When to use lists vs dictionaries vs tuples vs sets
2. How to efficiently access and modify data in each structure
3. List comprehensions and when they're better than loops
4. Dictionary methods and common patterns
5. How to work with nested data structures
6. Best practices for data organization

Please:
- Explain each data structure with real-world analogies
- Show me practical examples for each use case
- Help me understand the performance differences
- Give me exercises that combine multiple data structures
- Walk me through common data processing patterns
- Explain when to choose one structure over another

I want to build programs that can handle real data effectively. Please provide hands-on examples and let me practice each concept."

## Key Takeaways
- Lists are for ordered, changeable collections
- Dictionaries are for key-value relationships
- Tuples are for fixed, ordered data
- Sets are for unique, unordered collections
- Choose the right structure for your data
- Practice with real-world scenarios

## Next Steps
Master these data structures and you'll be ready for:
- File handling and data persistence
- Object-oriented programming
- Working with APIs and external data
- Building more complex applications

---

**Next Lesson**: [File Handling and Modules](04_file_handling_and_modules.md)
