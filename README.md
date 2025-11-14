# Machine Learning: Basic to Pro

A comprehensive guide to learning machine learning concepts from fundamentals to advanced techniques, with practical implementations and real-world examples.

## 📋 Table of Contents

- [Overview](#overview)
- [Contents](#contents)
- [Dataset](#dataset)
- [Topics Covered](#topics-covered)
- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Learning Path](#learning-path)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository contains a structured learning journey through machine learning concepts, starting from basic data cleaning and preprocessing techniques to advanced feature engineering and model optimization. The materials are presented through interactive Jupyter notebooks with detailed explanations and hands-on exercises.

## Contents

- **`dataset.ipynb`** - Main notebook containing comprehensive lessons on:
  - Data types and variables
  - Data cleaning and preprocessing
  - Handling missing values
  - Outlier detection and removal
  - Categorical encoding techniques
  - Feature scaling and transformation

- **`exams.csv`** - Sample dataset used for demonstrations and practice exercises

## Dataset

The `exams.csv` file contains student exam records with various attributes useful for practicing:
- Data exploration
- Missing value handling
- Categorical encoding
- Outlier detection
- Statistical analysis

## Topics Covered

### Data Types
- **Numerical Variables**
  - Discrete: Repeated categorical values
  - Continuous: Range-based values
- **Categorical Variables**
  - Ordinal: Sequential/ordered categories
  - Nominal: Unordered categories
- **Datetime & Mixed Variables**

### Data Cleaning Pipeline
1. **Handle Missing Values**
   - Detection using `.isnull()`
   - Filling strategies (forward fill, backward fill, mean, mode)
   - Removal when necessary

2. **Outlier Detection**
   - IQR (Interquartile Range) method
   - Z-score method
   - Visual detection with box plots and distribution plots

3. **Data Scaling & Transformation**
   - Normalization techniques
   - Standardization

4. **Encoding Categorical Data**
   - One-Hot Encoding (for binary/multi-class nominal data)
   - Label Encoding (for nominal data)
   - Ordinal Encoding (for ordered categories)

5. **Handle Duplicates**
   - Detection and removal

6. **Handle Inconsistent Data**
   - Data validation and correction

### Encoding Techniques

- **One-Hot Encoding**: Converts categories into binary columns
- **Label Encoding**: Maps categories to integer labels
- **Ordinal Encoding**: Encodes ordered categories with custom mapping

## Getting Started

### Requirements

- Python 3.7+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

### Installation

1. Clone the repository:
```bash
git clone https://github.com/neeteshdixit/Machine_learning-basic-to-pro.git
cd Machine_learning-basic-to-pro
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `dataset.ipynb` and start learning!

## Usage

### Running the Notebook

1. Navigate to the repository directory
2. Start Jupyter: `jupyter notebook`
3. Open `dataset.ipynb`
4. Execute cells sequentially to understand each concept
5. Modify code and experiment with the dataset

### Practicing with Your Own Data

The techniques demonstrated in this notebook can be applied to any CSV dataset:

```python
import pandas as pd

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Apply cleaning techniques
data.dropna(inplace=True)
# ... apply other preprocessing steps
```

## Learning Path

### Beginner Level
1. Understand different data types
2. Load and explore the dataset
3. Identify missing values
4. Learn basic data cleaning

### Intermediate Level
1. Master handling missing values with different strategies
2. Detect and understand outliers
3. Apply basic encoding techniques
4. Understand data distribution

### Advanced Level
1. Implement multiple encoding strategies
2. Optimize data preprocessing pipelines
3. Handle complex edge cases
4. Prepare data for machine learning models

## Key Functions Reference

### Missing Value Handling
```python
# Check for missing values
data.isnull().sum()

# Drop missing values
data.dropna(inplace=True)

# Fill with mean (numerical)
data['column'].fillna(data['column'].mean(), inplace=True)

# Fill with mode (categorical)
data['column'].fillna(data['column'].mode()[0], inplace=True)
```

### Outlier Detection (IQR Method)
```python
q1 = data["column"].quantile(0.25)
q3 = data["column"].quantile(0.75)
IQR = q3 - q1
lower_bound = q1 - 1.5 * IQR
upper_bound = q3 + 1.5 * IQR
```

### Encoding
```python
# One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
encoded = ohe.fit_transform(data[['column']])

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['column'] = le.fit_transform(data['column'])

# Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['small', 'medium', 'large']])
```

## Contributing

Contributions are welcome! Feel free to:
- Add more examples
- Improve explanations
- Fix any errors
- Add new topics

Please create a pull request with your improvements.

## License

This project is open source and available under the MIT License.

## Author

**Neetesh Dixit**
- Email: dixitneetesh857@gmail.com
- GitHub: [@neeteshdixit](https://github.com/neeteshdixit)

## Acknowledgments

This material is designed for learners at all levels who want to master machine learning fundamentals and best practices.

---

Happy Learning! 🚀

For questions or feedback, please open an issue on GitHub.
