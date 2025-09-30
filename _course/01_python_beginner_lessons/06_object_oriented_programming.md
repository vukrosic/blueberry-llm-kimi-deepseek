# Object-Oriented Programming in Python

## Learning Objectives
- Understand classes and objects
- Master inheritance and polymorphism
- Learn about encapsulation and abstraction
- Practice with real-world OOP examples

## What is Object-Oriented Programming?
Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects that contain both data (attributes) and behavior (methods). Think of it like creating blueprints for real-world things.

## Classes and Objects

### Creating a Simple Class
```python
class Dog:
    def __init__(self, name, age, breed):
        self.name = name
        self.age = age
        self.breed = breed
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def get_info(self):
        return f"{self.name} is a {self.age}-year-old {self.breed}"

# Creating objects (instances)
dog1 = Dog("Buddy", 3, "Golden Retriever")
dog2 = Dog("Max", 5, "German Shepherd")

# Using the objects
print(dog1.bark())  # Buddy says Woof!
print(dog2.get_info())  # Max is a 5-year-old German Shepherd
```

### Understanding `self`
```python
class Person:
    def __init__(self, name, age):
        self.name = name  # self refers to this specific instance
        self.age = age
    
    def introduce(self):
        return f"Hi, I'm {self.name} and I'm {self.age} years old"
    
    def have_birthday(self):
        self.age += 1
        return f"Happy birthday! Now I'm {self.age}"

# Creating instances
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Each instance has its own data
print(person1.introduce())  # Hi, I'm Alice and I'm 25 years old
print(person2.introduce())  # Hi, I'm Bob and I'm 30 years old

person1.have_birthday()
print(person1.introduce())  # Hi, I'm Alice and I'm 26 years old
```

## Class Attributes vs Instance Attributes

### Instance Attributes
```python
class Car:
    def __init__(self, make, model, year):
        self.make = make      # Instance attribute
        self.model = model    # Instance attribute
        self.year = year      # Instance attribute
        self.mileage = 0      # Instance attribute
    
    def drive(self, miles):
        self.mileage += miles
        return f"Drove {miles} miles. Total mileage: {self.mileage}"

# Each car has its own attributes
car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2021)

car1.drive(100)
print(car1.mileage)  # 100
print(car2.mileage)  # 0
```

### Class Attributes
```python
class Car:
    # Class attribute (shared by all instances)
    wheels = 4
    engine_type = "Internal Combustion"
    
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
        self.mileage = 0
    
    def get_specs(self):
        return f"{self.make} {self.model} has {Car.wheels} wheels"

# All cars share class attributes
car1 = Car("Toyota", "Camry", 2020)
car2 = Car("Honda", "Civic", 2021)

print(car1.get_specs())  # Toyota Camry has 4 wheels
print(car2.get_specs())  # Honda Civic has 4 wheels
print(Car.wheels)        # 4
```

## Inheritance

### Basic Inheritance
```python
class Animal:
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        return "Some generic animal sound"
    
    def get_info(self):
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Dog")  # Call parent constructor
        self.breed = breed
    
    def make_sound(self):  # Override parent method
        return "Woof!"
    
    def get_info(self):  # Override parent method
        return f"{self.name} is a {self.breed} dog"

class Cat(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Cat")
        self.breed = breed
    
    def make_sound(self):
        return "Meow!"

# Using inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", "Persian")

print(dog.get_info())    # Buddy is a Golden Retriever dog
print(cat.get_info())    # Whiskers is a Cat
print(dog.make_sound())  # Woof!
print(cat.make_sound())  # Meow!
```

### Multiple Inheritance
```python
class Flyable:
    def fly(self):
        return "Flying through the air"

class Swimmable:
    def swim(self):
        return "Swimming in water"

class Duck(Animal, Flyable, Swimmable):
    def __init__(self, name):
        super().__init__(name, "Duck")
    
    def make_sound(self):
        return "Quack!"

# Duck inherits from multiple classes
duck = Duck("Donald")
print(duck.make_sound())  # Quack!
print(duck.fly())         # Flying through the air
print(duck.swim())        # Swimming in water
```

## Encapsulation

### Private Attributes and Methods
```python
class BankAccount:
    def __init__(self, account_number, initial_balance):
        self.account_number = account_number
        self.__balance = initial_balance  # Private attribute
        self.__transaction_history = []   # Private attribute
    
    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            self.__add_transaction(f"Deposit: +${amount}")
            return f"Deposited ${amount}. New balance: ${self.__balance}"
        else:
            return "Invalid deposit amount"
    
    def withdraw(self, amount):
        if 0 < amount <= self.__balance:
            self.__balance -= amount
            self.__add_transaction(f"Withdrawal: -${amount}")
            return f"Withdrew ${amount}. New balance: ${self.__balance}"
        else:
            return "Insufficient funds or invalid amount"
    
    def get_balance(self):
        return self.__balance
    
    def __add_transaction(self, transaction):  # Private method
        self.__transaction_history.append(transaction)
    
    def get_transaction_history(self):
        return self.__transaction_history.copy()

# Using encapsulation
account = BankAccount("12345", 1000)
print(account.deposit(500))    # Deposited $500. New balance: $1500
print(account.withdraw(200))   # Withdrew $200. New balance: $1300
print(account.get_balance())   # 1300
# print(account.__balance)     # This would cause an error
```

### Property Decorators
```python
class Temperature:
    def __init__(self, celsius=0):
        self._celsius = celsius
    
    @property
    def celsius(self):
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        if value < -273.15:
            raise ValueError("Temperature cannot be below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        self.celsius = (value - 32) * 5/9

# Using properties
temp = Temperature(25)
print(f"Celsius: {temp.celsius}")      # Celsius: 25
print(f"Fahrenheit: {temp.fahrenheit}") # Fahrenheit: 77.0

temp.fahrenheit = 86
print(f"Celsius: {temp.celsius}")      # Celsius: 30.0
```

## Polymorphism

### Method Overriding
```python
class Shape:
    def area(self):
        return "Area calculation not implemented"
    
    def perimeter(self):
        return "Perimeter calculation not implemented"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Polymorphism in action
shapes = [
    Rectangle(5, 3),
    Circle(4),
    Rectangle(2, 8)
]

for shape in shapes:
    print(f"Area: {shape.area()}, Perimeter: {shape.perimeter()}")
```

## Practice Exercises

### Exercise 1: Library Management System
Create a library system with:
- Book class (title, author, ISBN, available)
- Library class (collection of books)
- Methods to add, remove, search, and borrow books
- Track borrowing history

### Exercise 2: Bank Account System
Build a banking system with:
- Account class (balance, account type)
- SavingsAccount and CheckingAccount classes
- Different interest rates and fees
- Transaction history and statements

### Exercise 3: Vehicle Hierarchy
Create a vehicle system with:
- Vehicle base class
- Car, Truck, Motorcycle subclasses
- Different fuel types and capacities
- Calculate fuel efficiency and range

### Exercise 4: Employee Management
Design an employee system with:
- Employee base class
- Manager, Developer, Designer subclasses
- Different salary calculations and benefits
- Performance tracking and promotions

## Common OOP Patterns

### Singleton Pattern
```python
class DatabaseConnection:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.connection_string = "database://localhost"
            self.initialized = True

# Only one instance will be created
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # True
```

### Factory Pattern
```python
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type, name):
        if animal_type == "dog":
            return Dog(name, "Mixed")
        elif animal_type == "cat":
            return Cat(name, "Mixed")
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

# Using the factory
dog = AnimalFactory.create_animal("dog", "Buddy")
cat = AnimalFactory.create_animal("cat", "Whiskers")
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm learning Object-Oriented Programming in Python. I understand basic Python but I'm struggling with:

1. Understanding classes vs objects and when to use them
2. How inheritance works and when to use it
3. The difference between class and instance attributes
4. Encapsulation and when to make attributes private
5. Polymorphism and method overriding
6. How to design good class hierarchies

Please:
- Explain OOP concepts with real-world analogies
- Show me step-by-step examples of creating classes
- Help me understand when to use inheritance vs composition
- Walk me through designing a complete OOP system
- Give me exercises that build progressively complex systems
- Explain best practices and common patterns

I want to write well-structured, maintainable code using OOP principles. Please provide practical examples and help me think like an object-oriented programmer."

## Key Takeaways
- Classes are blueprints, objects are instances
- Inheritance promotes code reuse
- Encapsulation protects data integrity
- Polymorphism allows flexible code
- Design classes based on real-world entities
- Practice with progressively complex examples

## Next Steps
Master OOP and you'll be ready for:
- Advanced Python features
- Design patterns
- Building complex applications
- Working with frameworks
- Preparing for AI/ML development

---

**Next Lesson**: [Advanced Python Features](07_advanced_python_features.md)
