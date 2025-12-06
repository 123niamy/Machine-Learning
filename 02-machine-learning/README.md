# Machine Learning

Classical machine learning algorithms, model training, evaluation, and predictive analytics.

## Projects

### Mathematical model1.py
**Logistic Regression Classification Model** for Titanic survival prediction.

**What it does:**
- Predicts passenger survival based on age, sex, and class
- Implements the complete ML workflow (8 steps)
- Evaluates model performance with accuracy and confusion matrix

**Key Features:**
- Data preprocessing with missing value handling
- Categorical encoding using one-hot encoding (pd.get_dummies)
- Train/test split for proper validation
- Logistic regression with extended iterations for convergence
- Performance metrics: accuracy score and confusion matrix

**Tech Stack:**
- scikit-learn for ML algorithms and preprocessing
- pandas for data manipulation
- seaborn for dataset loading

**Usage:**
```bash
python "Mathematical model1.py"
```

**Output:**
```
Accuracy: ~0.80 (80% correct predictions)
Confusion Matrix:
[[139  20]
 [ 34  74]]
```

---

## ML Workflow Demonstrated

This project follows the **standard 8-step ML workflow**:

1. **Data Collection** - Load Titanic dataset
2. **Data Preparation** - Select features and target variable
3. **Data Preprocessing** - Handle missing values, encode categories
4. **Data Splitting** - 70% train, 30% test
5. **Model Selection** - Logistic Regression for binary classification
6. **Model Training** - Fit model to training data
7. **Prediction** - Make predictions on test set
8. **Evaluation** - Calculate accuracy and confusion matrix

---

## Concepts Covered

- **Binary Classification**: Predicting yes/no outcomes (survived/died)
- **Feature Selection**: Choosing relevant predictors
- **One-Hot Encoding**: Converting categorical variables to numeric
- **Train-Test Split**: Preventing overfitting with held-out data
- **Logistic Regression**: Linear model for classification
- **Model Evaluation**: Accuracy, confusion matrix interpretation

---

**Coming Soon:**
- Decision trees and random forests
- Support vector machines (SVM)
- Cross-validation and hyperparameter tuning
- Feature engineering and selection
- Ensemble methods
