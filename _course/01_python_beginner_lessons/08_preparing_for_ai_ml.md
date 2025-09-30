# Preparing for AI/ML Development

## Learning Objectives
- Understand the Python ecosystem for AI/ML
- Learn essential libraries and tools
- Master data manipulation and visualization
- Practice with real AI/ML examples

## Python Libraries for AI/ML

### Essential Libraries
```python
# Data manipulation and analysis
import pandas as pd
import numpy as np

# Machine learning
from sklearn import datasets, model_selection, metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Deep learning
import torch
import tensorflow as tf

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
import scipy.stats as stats
```

### Installing Libraries
```bash
# Install essential packages
pip install pandas numpy scikit-learn matplotlib seaborn

# For deep learning
pip install torch tensorflow

# For data science
pip install jupyter notebook

# For specific AI tasks
pip install opencv-python pillow transformers
```

## NumPy - Numerical Computing

### Basic NumPy Operations
```python
import numpy as np

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

# Array properties
print(arr1.shape)    # (5,)
print(arr2.shape)    # (2, 3)
print(arr1.dtype)    # int64

# Array operations
result = arr1 * 2    # [2, 4, 6, 8, 10]
sum_result = np.sum(arr1)  # 15
mean_result = np.mean(arr1)  # 3.0

# Array creation
zeros = np.zeros((3, 4))      # 3x4 array of zeros
ones = np.ones((2, 3))        # 2x3 array of ones
random_arr = np.random.rand(3, 3)  # 3x3 random array
```

### NumPy for Data Processing
```python
# Reshaping arrays
arr = np.arange(12)
reshaped = arr.reshape(3, 4)

# Array indexing and slicing
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
first_row = data[0, :]        # [1, 2, 3]
first_col = data[:, 0]        # [1, 4, 7]
subarray = data[1:3, 1:3]     # [[5, 6], [8, 9]]

# Mathematical operations
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])

dot_product = np.dot(x, y)    # 110
matrix_mult = np.matmul(x.reshape(1, -1), y.reshape(-1, 1))
```

## Pandas - Data Manipulation

### Working with DataFrames
```python
import pandas as pd

# Creating DataFrames
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['IT', 'HR', 'IT', 'Finance']
}

df = pd.DataFrame(data)
print(df)

# Basic operations
print(df.head())           # First 5 rows
print(df.info())           # DataFrame info
print(df.describe())       # Statistical summary
print(df['Age'].mean())    # Average age
```

### Data Manipulation
```python
# Selecting data
young_employees = df[df['Age'] < 30]
it_department = df[df['Department'] == 'IT']
high_salary = df[df['Salary'] > 55000]

# Adding columns
df['Bonus'] = df['Salary'] * 0.1
df['Age_Group'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Senior')

# Grouping and aggregation
dept_stats = df.groupby('Department').agg({
    'Age': 'mean',
    'Salary': ['mean', 'sum', 'count']
})

# Sorting
sorted_df = df.sort_values('Salary', ascending=False)
```

### Handling Missing Data
```python
# Creating sample data with missing values
data_with_nulls = {
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, 4, None],
    'C': [1, 2, 3, 4, 5]
}

df_nulls = pd.DataFrame(data_with_nulls)

# Check for missing values
print(df_nulls.isnull().sum())

# Handle missing values
df_filled = df_nulls.fillna(0)                    # Fill with 0
df_dropped = df_nulls.dropna()                    # Drop rows with nulls
df_forward = df_nulls.fillna(method='ffill')      # Forward fill
```

## Matplotlib - Data Visualization

### Basic Plotting
```python
import matplotlib.pyplot as plt
import numpy as np

# Line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
x = np.random.randn(100)
y = 2 * x + np.random.randn(100)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.show()
```

### Multiple Plots
```python
# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Line plot
axes[0, 0].plot(x, y)
axes[0, 0].set_title('Line Plot')

# Plot 2: Histogram
axes[0, 1].hist(y, bins=20, alpha=0.7)
axes[0, 1].set_title('Histogram')

# Plot 3: Bar plot
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('Bar Plot')

# Plot 4: Box plot
data_box = [np.random.normal(0, std, 100) for std in range(1, 4)]
axes[1, 1].boxplot(data_box)
axes[1, 1].set_title('Box Plot')

plt.tight_layout()
plt.show()
```

## Scikit-learn - Machine Learning

### Basic Machine Learning Pipeline
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
```

### Regression Example
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()
```

## Practice Exercises

### Exercise 1: Data Analysis Project
Create a data analysis project that:
- Loads a dataset (CSV file)
- Performs exploratory data analysis
- Creates visualizations
- Identifies patterns and insights
- Generates a summary report

### Exercise 2: Simple ML Model
Build a machine learning model that:
- Uses a real dataset
- Implements data preprocessing
- Trains multiple models
- Compares model performance
- Visualizes results

### Exercise 3: Data Pipeline
Create a data processing pipeline that:
- Reads data from multiple sources
- Cleans and transforms data
- Handles missing values
- Exports processed data
- Includes error handling

### Exercise 4: Interactive Dashboard
Build an interactive data dashboard that:
- Displays key metrics
- Allows filtering and sorting
- Updates in real-time
- Exports data and charts
- Is user-friendly

## Common AI/ML Patterns

### Data Preprocessing Pipeline
```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(X_train, X_test, y_train=None):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    return X_train_scaled, X_test_scaled, scaler
```

### Model Evaluation
```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score

def evaluate_model(model, X, y):
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.2f}")
    
    # Confusion matrix
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    # ROC AUC (for binary classification)
    if len(np.unique(y)) == 2:
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        print(f"ROC AUC: {auc:.2f}")
```

## AI Learning Prompt

**Copy this prompt into ChatGPT or any AI chatbot:**

"I'm preparing for AI/ML development and need help with Python libraries and concepts. I understand basic Python but I'm struggling with:

1. How to use NumPy for numerical computing
2. Pandas for data manipulation and analysis
3. Matplotlib for data visualization
4. Scikit-learn for machine learning
5. Best practices for data preprocessing
6. How to build complete ML pipelines

Please:
- Explain each library with practical examples
- Show me how to work with real datasets
- Help me understand data preprocessing steps
- Walk me through building ML models
- Give me exercises with real-world data
- Explain common pitfalls and best practices

I want to build practical AI/ML applications. Please provide hands-on examples and help me think like a data scientist."

## Key Takeaways
- NumPy is essential for numerical computing
- Pandas makes data manipulation easy
- Matplotlib enables effective data visualization
- Scikit-learn provides powerful ML tools
- Data preprocessing is crucial for good results
- Practice with real datasets builds confidence

## Next Steps
Master these AI/ML foundations and you'll be ready for:
- Deep learning with PyTorch/TensorFlow
- Advanced ML algorithms
- Building production ML systems
- Working with large datasets
- Contributing to AI research

---

**Next Lesson**: [Python Best Practices](09_python_best_practices.md)
