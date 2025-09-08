# Water Quality Prediction

This project is a simple machine learning implementation that predicts **water potability** using several classification algorithms, including:
- Logistic Regression
- Decision Tree Classifier
- Support Vector Machine (SVM)

---

## Dataset

The dataset used contains physicochemical properties of water samples along with a binary label indicating whether the water is **potable** (safe to drink) or not.

**Features include:**
- `ph`: pH value of water
- `Hardness`: Calcium and magnesium content
- `Solids`: Total dissolved solids
- `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- `Potability`: Target variable (0 = Not potable, 1 = Potable)

---

## Setup

### Requirements

Install the required Python packages with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
````

---

## Usage

Run the Python script:

```bash
python water_quality_prediction.py
```

### What the script does:

1. Loads the dataset from a remote CSV URL from Kaggle.
2. Explores and prints data properties (shape, types, nulls).
3. Handles missing values by imputing mean values to clean dataset.
4. Splits the data into features (`X`) and labels (`y`).
5. Splits the data further into training and testing sets.
6. Scales the features using `StandardScaler`.
7. Trains three classification models:
   * Logistic Regression
   * Decision Tree
   * SVM (Support Vector Classifier)
8. Evaluates the models using accuracy scores and classification reports.

---

## Models Used

| Model               | Description                            |
| ------------------- | -------------------------------------- |
| Logistic Regression | Linear model for binary classification |
| Decision Tree       | Tree-based model for classification    |
| SVM                 | Finds optimal separating hyperplane    |

---

## Results & Evaluation

This project aims to use several machine learning models to predict the potability of water (the target variable in the dataset). The script prints information about the dataset as well as the accuracy of each model after training on the dataset. Accuracy is calculated using the test set from a 67-33 train-test split, and this metric is used to compare and evaluate the performance of the three models. Out of the three, the Support Vector Machine (SVM) resulted in the highest accuracy, the Logistic Regression model resulted in the second highest accuracy, and the Decision Tree model resulted in the lowest accuracy. The SVM model performed the best with the highest number of correct predictions for water potability while the Decision Tree performed the worst because it resulted with the lowest number of correct predictions. Based on these results, the Logistic Regression model is most suited for this classification task whereas the Decision Tree model is the least suited.

---

## Notes

* Missing values in `ph`, `Sulfate`, and `Trihalomethanes` are filled using the column mean.
* Feature scaling is done using `StandardScaler`.
* No hyperparameter tuning or cross-validation is performed.
* The dataset is accessed from a temporary Kaggle download link.

---

## Reference

[Kaggle Notebook](https://www.kaggle.com/code/imakash3011/water-quality-prediction-7-model)