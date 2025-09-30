# Control Flow and Loops in Python

## Learning Objectives
- Master if statements and conditional logic
- Understand for loops and while loops
- Learn about break and continue statements
- Practice with real-world examples

## What is Control Flow?
Control flow determines the order in which your code executes. It's like giving your program the ability to make decisions and repeat actions.

## If Statements (Conditional Logic)

### Basic If Statement
```python
age = 18

if age >= 18:
    print("You are an adult")
else:
    print("You are a minor")
```

### Multiple Conditions
```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Your grade is: {grade}")
```

### Logical Operators
```python
# AND operator - both conditions must be true
age = 25
has_license = True

if age >= 18 and has_license:
    print("You can drive")

# OR operator - either condition can be true
weather = "sunny"
temperature = 75

if weather == "sunny" or temperature > 70:
    print("Great day for a walk")

# NOT operator - reverses the condition
is_weekend = False

if not is_weekend:
    print("It's a weekday")
```

## For Loops

### Basic For Loop
```python
# Loop through a list
fruits = ["apple", "banana", "orange"]

for fruit in fruits:
    print(f"I like {fruit}")
```

### Range Function
```python
# Count from 0 to 4
for i in range(5):
    print(i)

# Count from 1 to 10
for i in range(1, 11):
    print(i)

# Count by 2s from 0 to 10
for i in range(0, 11, 2):
    print(i)
```

### Loop with Index
```python
fruits = ["apple", "banana", "orange"]

for index, fruit in enumerate(fruits):
    print(f"{index + 1}. {fruit}")
```

## While Loops

### Basic While Loop
```python
count = 0

while count < 5:
    print(f"Count: {count}")
    count += 1
```

### User Input Loop
```python
user_input = ""

while user_input.lower() != "quit":
    user_input = input("Enter something (or 'quit' to exit): ")
    print(f"You entered: {user_input}")
```

## Break and Continue

### Break Statement
```python
# Break out of loop early
for i in range(10):
    if i == 5:
        break
    print(i)
# Output: 0, 1, 2, 3, 4
```

### Continue Statement
```python
# Skip current iteration
for i in range(5):
    if i == 2:
        continue
    print(i)
# Output: 0, 1, 3, 4
```

## Practice Exercises

### Exercise 1: Grade Calculator
Write a program that:
- Takes a student's score as input
- Assigns a letter grade (A, B, C, D, F)
- Prints the grade and a message

### Exercise 2: Number Guessing Game
Create a simple guessing game:
- Computer picks a random number 1-10
- User tries to guess it
- Give hints (too high/too low)
- Count the number of attempts

### Exercise 3: FizzBuzz
Write a program that prints numbers 1-20, but:
- For multiples of 3, print "Fizz"
- For multiples of 5, print "Buzz"
- For multiples of both 3 and 5, print "FizzBuzz"

### Exercise 4: Shopping List
Create a program that:
- Lets users add items to a shopping list
- Shows the current list
- Allows removing items
- Continues until user types "done"

## Common Patterns

### Input Validation
```python
while True:
    try:
        age = int(input("Enter your age: "))
        if age > 0:
            break
        else:
            print("Age must be positive")
    except ValueError:
        print("Please enter a valid number")
```

### Menu System
```python
while True:
    print("\n1. Add item")
    print("2. View list")
    print("3. Remove item")
    print("4. Exit")
    
    choice = input("Choose an option: ")
    
    if choice == "1":
        # Add item logic
        pass
    elif choice == "2":
        # View list logic
        pass
    elif choice == "3":
        # Remove item logic
        pass
    elif choice == "4":
        print("Goodbye!")
        break
    else:
        print("Invalid choice")
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning control flow and loops in Python. I understand basic variables and functions, but I'm struggling with:

1. When to use if vs elif vs else statements
2. How for loops work with lists and ranges
3. The difference between for loops and while loops
4. When to use break and continue
5. How to create interactive programs with user input

Please:
- Explain each concept with simple analogies
- Show me step-by-step examples
- Help me understand the logic behind each decision
- Give me practice problems with solutions
- Walk me through debugging when I make mistakes
- Use real-world examples I can relate to

I want to build confidence in writing programs that can make decisions and repeat actions. Please be patient and let me practice each concept thoroughly."

## Key Takeaways
- If statements help your program make decisions
- For loops are great for known iterations
- While loops are perfect for unknown iterations
- Break and continue give you more control
- Practice with real problems builds confidence
- Start simple and add complexity gradually

## Next Steps
Master these concepts and you'll be ready for:
- Lists and data structures
- Functions with parameters
- Error handling
- Object-oriented programming
- Building interactive applications

---

**Next Lesson**: [Lists and Data Structures](03_lists_and_data_structures.md)
