# 🚢 Titanic Survival Prediction Using Decision Tree

## 📌 Project Overview

This project uses a **Decision Tree Classifier** to predict whether a Titanic passenger survived or did not survive based on passenger information.

The project demonstrates a complete beginner-friendly machine learning workflow, including:

* Data loading
* Data exploration
* Handling missing values
* Categorical data encoding
* Train-test splitting
* Decision Tree model training
* Model prediction
* Model evaluation
* Confusion matrix
* Decision Tree visualization

---

## 🎯 Objective

The objective of this project is to build a machine learning classification model that predicts:

```text
0 → Did Not Survive
1 → Survived
```

based on information about the passenger.

---

## 📊 Dataset

The dataset contains information about Titanic passengers.

### Features

| Feature       | Description                            |
| ------------- | -------------------------------------- |
| `PassengerId` | Unique passenger identification number |
| `Pclass`      | Passenger class                        |
| `Sex`         | Passenger gender                       |
| `Age`         | Passenger age                          |
| `SibSp`       | Number of siblings/spouses aboard      |
| `Parch`       | Number of parents/children aboard      |
| `Fare`        | Passenger ticket fare                  |
| `Embarked`    | Port of embarkation                    |
| `Survived`    | Target variable                        |

### Target

```text
Survived
```

* `0` = Did not survive
* `1` = Survived

---

## 🧠 Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree is a **supervised machine learning algorithm** that makes predictions using a sequence of conditions or decisions.

For example, a tree may learn rules conceptually like:

```text
Pclass <= 2?
   ↓
Sex = Female?
   ↓
Age <= 30?
   ↓
Prediction
```

The model automatically learns which features and conditions provide useful splits in the training data.

### Why Decision Tree?

Decision Tree was selected because:

* It can handle non-linear relationships.
* It can work with numerical and encoded categorical features.
* It is easy to understand and visualize.
* Feature scaling is not normally required.
* Its decision-making process can be represented as a tree of rules.

---

## 🔄 Project Workflow

```text
Titanic Dataset
       ↓
Load Dataset
       ↓
Explore Dataset
       ↓
Check Missing Values
       ↓
Handle Missing Values
       ↓
Encode Categorical Data
       ↓
Remove Passenger ID
       ↓
Separate Features and Target
       ↓
Train-Test Split
       ↓
Train Decision Tree
       ↓
Make Predictions
       ↓
Evaluate Model
       ↓
Confusion Matrix
       ↓
Visualize Decision Tree
```

---

## 🧹 Data Preprocessing

### Missing Values

The dataset contains missing values.

For numerical columns:

```python
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
```

Missing values are replaced using the median.

For the categorical `Embarked` column:

```python
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)
```

The most frequently occurring category is used.

### Categorical Encoding

Machine learning models require numerical input, so categorical columns are converted using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)
```

### Removing Passenger ID

`PassengerId` is removed because it is only an identifier and does not provide useful predictive information.

---

## ✂️ Train-Test Split

The dataset is divided into training and testing data:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Split

* **80%** → Training data
* **20%** → Testing data

The model learns patterns from the training data and is evaluated on unseen testing data.

---

## 🌳 Decision Tree Configuration

The model is created using:

```python
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)
```

### Parameters

**`criterion="gini"`**

Uses Gini impurity to determine useful splits.

**`max_depth=5`**

Limits the depth of the tree and helps control model complexity.

**`random_state=42`**

Makes the results reproducible.

---

## 📈 Model Evaluation

The model is evaluated using:

### Accuracy

Measures the proportion of all predictions that were correct.

### Precision

Measures how many of the passengers predicted as survivors actually survived.

### Recall

Measures how many of the passengers who actually survived were correctly identified.

### F1 Score

Provides a combined measure of precision and recall.

The project also uses a **confusion matrix** to understand correct and incorrect predictions.

---

## 🔲 Confusion Matrix

The confusion matrix contains four possible outcomes:

|               |   Predicted: 0 |   Predicted: 1 |
| ------------- | -------------: | -------------: |
| **Actual: 0** |  True Negative | False Positive |
| **Actual: 1** | False Negative |  True Positive |

This helps identify exactly where the model is making classification errors.

---

## 🌳 Decision Tree Visualization

The trained Decision Tree is visualized using:

```python
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[
        "Did Not Survive",
        "Survived"
    ],
    filled=True,
    rounded=True
)
```

This makes the learned decision structure easier to understand.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

---

## 📁 Project Structure

```text
titanic-survival-decision-tree/
│
├── titanic_with_null_values.csv
├── titanic_decision_tree.py
└── README.md
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd titanic-survival-decision-tree
```

### 3. Install dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 4. Run the program

```bash
python titanic_decision_tree.py
```

---

## 📌 Key Learning Outcomes

Through this project, I learned how to:

* Work with a CSV dataset using Pandas
* Identify and handle missing values
* Encode categorical variables
* Separate features and target variables
* Split data into training and testing sets
* Build a Decision Tree classification model
* Make predictions on unseen data
* Evaluate a classification model
* Interpret a confusion matrix
* Visualize a trained Decision Tree

---

## ⚠️ Note

The dataset used in this project is a **small practice dataset containing 50 records with intentionally added missing values**. Because of its small size, the evaluation metrics should be considered suitable for learning and demonstration rather than real-world performance.

---

## 👨‍💻 Project Purpose

This project was developed as part of my journey in learning **Machine Learning and Python**, with a focus on understanding the complete process of building and evaluating a classification model.

#MachineLearning #Python #DecisionTree #ScikitLearn #DataScience #MachineLearningProject
