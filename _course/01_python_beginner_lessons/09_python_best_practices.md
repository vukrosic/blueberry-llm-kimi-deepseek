# Python Best Practices and Code Quality

## Learning Objectives
- Learn Python coding standards and style
- Master testing and debugging techniques
- Understand performance optimization
- Practice with production-ready code

## Code Style and Standards

### PEP 8 - Python Style Guide
```python
# Good: Clear variable names
user_name = "Alice"
user_age = 25
is_active = True

# Bad: Unclear names
u = "Alice"
a = 25
flag = True

# Good: Function names
def calculate_total_price(items):
    return sum(item.price for item in items)

# Bad: Unclear function names
def calc(items):
    return sum(i.p for i in items)

# Good: Class names
class UserManager:
    pass

# Bad: Unclear class names
class um:
    pass
```

### Code Formatting
```python
# Good: Proper indentation and spacing
def process_data(data, threshold=0.5):
    """Process data with given threshold."""
    if not data:
        return []
    
    filtered_data = [
        item for item in data 
        if item.value > threshold
    ]
    
    return sorted(filtered_data, key=lambda x: x.priority)

# Bad: Poor formatting
def process_data(data,threshold=0.5):
    if not data:return []
    filtered_data=[item for item in data if item.value>threshold]
    return sorted(filtered_data,key=lambda x:x.priority)
```

### Docstrings and Comments
```python
def calculate_compound_interest(principal, rate, time, compound_frequency=12):
    """
    Calculate compound interest.
    
    Args:
        principal (float): Initial amount of money
        rate (float): Annual interest rate (as decimal)
        time (int): Time period in years
        compound_frequency (int): Number of times interest is compounded per year
    
    Returns:
        float: Final amount after compound interest
    
    Raises:
        ValueError: If any parameter is negative
    
    Example:
        >>> calculate_compound_interest(1000, 0.05, 2)
        1104.713067441311
    """
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("All parameters must be non-negative")
    
    # Calculate compound interest using the formula
    # A = P(1 + r/n)^(nt)
    amount = principal * (1 + rate / compound_frequency) ** (compound_frequency * time)
    
    return amount
```

## Error Handling Best Practices

### Specific Exception Handling
```python
# Good: Specific exceptions
def read_config_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Config file {filename} not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filename}: {e}")
        return {}
    except PermissionError:
        logger.error(f"Permission denied accessing {filename}")
        return {}

# Bad: Catching all exceptions
def read_config_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except Exception:
        return {}
```

### Custom Exceptions
```python
class ValidationError(Exception):
    """Raised when data validation fails."""
    pass

class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds."""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: {balance} < {amount}")

def withdraw(account, amount):
    if amount <= 0:
        raise ValidationError("Amount must be positive")
    
    if account.balance < amount:
        raise InsufficientFundsError(account.balance, amount)
    
    account.balance -= amount
    return account.balance
```

## Testing

### Unit Testing with unittest
```python
import unittest
from calculator import Calculator

class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.calc = Calculator()
    
    def test_add_positive_numbers(self):
        """Test addition of positive numbers."""
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_add_negative_numbers(self):
        """Test addition of negative numbers."""
        result = self.calc.add(-2, -3)
        self.assertEqual(result, -5)
    
    def test_divide_by_zero(self):
        """Test division by zero raises exception."""
        with self.assertRaises(ValueError):
            self.calc.divide(10, 0)
    
    def tearDown(self):
        """Clean up after each test method."""
        pass

if __name__ == '__main__':
    unittest.main()
```

### Testing with pytest
```python
import pytest
from calculator import Calculator

@pytest.fixture
def calculator():
    """Create a calculator instance for testing."""
    return Calculator()

def test_add_positive_numbers(calculator):
    """Test addition of positive numbers."""
    assert calculator.add(2, 3) == 5

def test_add_negative_numbers(calculator):
    """Test addition of negative numbers."""
    assert calculator.add(-2, -3) == -5

@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (-2, -3, -5),
    (0, 5, 5),
    (5, 0, 5)
])
def test_add_multiple_cases(calculator, a, b, expected):
    """Test addition with multiple test cases."""
    assert calculator.add(a, b) == expected

def test_divide_by_zero(calculator):
    """Test division by zero raises exception."""
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculator.divide(10, 0)
```

## Performance Optimization

### Efficient Data Structures
```python
# Good: Use appropriate data structures
from collections import defaultdict, Counter

# Counting occurrences
def count_words_efficient(text):
    return Counter(text.split())

# Grouping data
def group_by_category(items):
    groups = defaultdict(list)
    for item in items:
        groups[item.category].append(item)
    return dict(groups)

# Bad: Inefficient approaches
def count_words_inefficient(text):
    words = text.split()
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts
```

### List Comprehensions vs Loops
```python
# Good: List comprehension
squares = [x**2 for x in range(10) if x % 2 == 0]

# Good: Generator expression for large datasets
large_squares = (x**2 for x in range(1000000) if x % 2 == 0)

# Bad: Traditional loop
squares = []
for x in range(10):
    if x % 2 == 0:
        squares.append(x**2)
```

### Caching and Memoization
```python
from functools import lru_cache
import time

# Good: Using lru_cache for expensive computations
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Good: Manual caching for complex scenarios
class DataProcessor:
    def __init__(self):
        self._cache = {}
    
    def process_data(self, data_id):
        if data_id in self._cache:
            return self._cache[data_id]
        
        # Expensive computation
        result = self._expensive_computation(data_id)
        self._cache[data_id] = result
        return result
    
    def _expensive_computation(self, data_id):
        time.sleep(1)  # Simulate expensive operation
        return f"Processed {data_id}"
```

## Code Organization

### Project Structure
```
my_project/
├── README.md
├── requirements.txt
├── setup.py
├── tests/
│   ├── __init__.py
│   ├── test_calculator.py
│   └── test_utils.py
├── src/
│   ├── __init__.py
│   ├── calculator.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       └── user.py
└── docs/
    ├── api.md
    └── examples.md
```

### Module Organization
```python
# calculator.py
"""Calculator module for basic mathematical operations."""

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history.copy()

# utils.py
"""Utility functions for the application."""

def validate_number(value):
    """Validate if value is a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def format_result(result, precision=2):
    """Format result with specified precision."""
    return f"{result:.{precision}f}"
```

## Logging

### Proper Logging Setup
```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class UserService:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def create_user(self, username, email):
        self.logger.info(f"Creating user: {username}")
        
        try:
            # User creation logic
            user = self._create_user_in_db(username, email)
            self.logger.info(f"User created successfully: {user.id}")
            return user
        except Exception as e:
            self.logger.error(f"Failed to create user {username}: {e}")
            raise
    
    def _create_user_in_db(self, username, email):
        # Simulate database operation
        pass
```

## Practice Exercises

### Exercise 1: Code Review
Review and improve the following code:
```python
def process_users(users):
    result = []
    for user in users:
        if user.age > 18:
            if user.email:
                if '@' in user.email:
                    result.append(user.name.upper())
    return result
```

### Exercise 2: Write Tests
Create comprehensive tests for a simple banking system:
- Account creation and management
- Deposit and withdrawal operations
- Balance calculations
- Error handling

### Exercise 3: Performance Optimization
Optimize the following function:
```python
def find_duplicates(data):
    duplicates = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] == data[j]:
                duplicates.append(data[i])
    return duplicates
```

### Exercise 4: Logging Implementation
Add proper logging to a file processing system:
- File reading operations
- Data validation
- Error handling
- Performance monitoring

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning Python best practices and code quality. I understand basic Python but I need help with:

1. Writing clean, readable code following PEP 8
2. Proper error handling and exception management
3. Writing effective tests for my code
4. Performance optimization techniques
5. Code organization and project structure
6. Logging and debugging best practices

Please:
- Review my code and suggest improvements
- Help me write comprehensive tests
- Show me how to optimize performance
- Guide me through proper project organization
- Explain logging strategies and debugging techniques
- Give me exercises to practice best practices

I want to write production-ready, maintainable Python code. Please provide practical examples and help me develop good coding habits."

## Key Takeaways
- Follow PEP 8 for consistent code style
- Write clear, descriptive names and docstrings
- Handle errors gracefully with specific exceptions
- Write tests for all your code
- Optimize performance when necessary
- Organize code in logical modules and packages
- Use logging for debugging and monitoring

## Next Steps
Master these best practices and you'll be ready for:
- Building production applications
- Contributing to open source projects
- Working in professional development teams
- Creating maintainable, scalable code
- Preparing for technical interviews
