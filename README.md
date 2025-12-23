# Decision Tree Classification from Scratch and Using scikit-learn

## Project Overview
This project demonstrates **Decision Tree classification** implemented in two ways:

1. **From scratch (pure Python)** – to deeply understand how decision trees work internally using concepts such as:
   - Gini Impurity
   - Information Gain
   - Recursive tree building
2. **Using scikit-learn** – to apply a production-level machine learning workflow on a real dataset.

The project uses a simple **fruit classification problem** based on features like color and diameter, making it ideal for learning and academic demonstration purposes.

---

## Objectives
- Understand the internal mechanics of a Decision Tree classifier
- Implement core algorithms without external ML libraries
- Compare manual implementation with scikit-learn’s optimized version
- Visualize data and trained decision trees
- Evaluate model performance using standard metrics

---

## Technologies Used
- Python 3
- NumPy
- Pandas
- Matplotlib
- scikit-learn

---

## Dataset Description
Two datasets are used:

### 1. Manual Dataset (From Scratch)
A small in-memory dataset:
- Features: Color, Diameter
- Labels: Apple, Grape, Lemon

Example:
- Green, 3 → Apple
- Red, 1 → Grape
- Yellow, 3 → Lemon

### 2. CSV Dataset (scikit-learn)
- Loaded from a CSV file (`fruit.csv`)
- Features:
  - Color (categorical)
  - Diameter (numeric)
- Label:
  - Fruit type

---

## Project Structure
- Decision Tree implemented from scratch using Python
- Data visualization using Matplotlib
- Feature encoding using OneHotEncoder
- Model training, testing, and evaluation using scikit-learn
- Tree visualization for interpretability

---

## Decision Tree from Scratch – Key Concepts

### Implemented Components
- **Gini Impurity**: Measures dataset impurity
- **Information Gain**: Determines best feature split
- **Recursive Tree Building**: Constructs tree until no gain remains
- **Question Nodes**: Handle numeric and categorical comparisons
- **Leaf Nodes**: Store class predictions
- **Tree Traversal**: Used for prediction/classification

### Core Algorithms
- Data partitioning based on feature thresholds
- Best split selection using maximum information gain
- Recursive construction of true/false branches
- Prediction using tree traversal

### Output
- Printed tree structure
- Prediction probabilities for test samples

---

## Decision Tree using scikit-learn – Workflow

### Steps Performed
1. Load dataset from CSV using Pandas
2. Visualize feature relationships using scatter plots
3. Encode categorical features using OneHotEncoder
4. Split dataset into training and testing sets
5. Train `DecisionTreeClassifier`
6. Make predictions on test data
7. Evaluate using:
   - Accuracy Score
   - Classification Report
8. Visualize the trained decision tree

---

## Model Evaluation
- Accuracy score printed for test set
- Precision, recall, and F1-score shown via classification report
- Prediction probabilities generated using `predict_proba`

---

## Visualization
- Scatter plots showing feature distribution
- Dual-axis visualization for color and diameter
- Fully rendered decision tree with:
  - Feature names
  - Class names
  - Color-coded nodes

---

## How to Run the Project

### Prerequisites
Install required libraries:
- numpy
- pandas
- matplotlib
- scikit-learn

### Execution
1. Run the Python script
2. Ensure the CSV file path is correct
3. Observe:
   - Printed decision tree
   - Model accuracy and metrics
   - Visual plots and decision tree diagram

---

## Learning Outcomes
- Clear understanding of how decision trees work internally
- Experience implementing ML algorithms without libraries
- Hands-on comparison between theory and real-world ML tools
- Improved understanding of feature encoding and model evaluation

---

## Future Improvements
- Extend scratch implementation to support pruning
- Add entropy-based splitting
- Support larger and continuous datasets
- Export trained models
- Integrate cross-validation

---

## Use Case
This project is suitable for:
- Machine Learning beginners
- Academic coursework
- Interview preparation
- Demonstrating core ML concepts in portfolios

---

## Author
Developed as an educational and experimental project to understand and apply Decision Tree Classification both theoretically and practically.
