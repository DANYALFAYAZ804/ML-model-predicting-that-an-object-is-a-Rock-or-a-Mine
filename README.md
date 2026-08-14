# Sonar Rock vs. Mine Classification

A machine learning project that classifies sonar signal readings as either a **Rock (R)** or a **Mine (M)**, using a Logistic Regression model built with scikit-learn. The model is trained on the classic Sonar dataset, which contains sonar frequency-response readings bounced off two types of underwater surfaces: metal cylinders (mines) and rocks.

## 📌 Overview

Sonar systems work by emitting sound waves and measuring the energy of the signal that bounces back. Different materials and surfaces reflect this energy differently across a range of frequencies. This project leverages that principle to build a supervised binary classifier capable of predicting whether an object detected by sonar is a **rock** or a **mine** — a task with real-world relevance to naval mine detection and underwater safety systems.

## 📊 Dataset

- **Source:** `sonar data.csv`
- **Samples:** 208 rows
- **Features:** 60 numerical columns (0–59), each representing sonar energy readings within a specific frequency band, normalized between 0.0 and 1.0
- **Label:** Column `60`, containing either:
  - `R` — Rock
  - `M` — Mine
- No column headers; the dataset is loaded with `header=None`

## ⚙️ Project Workflow

1. **Data Loading & Exploration**
   - Load the dataset using `pandas`
   - Inspect structure with `.head()`, `.shape`, and `.describe()`
   - Check class distribution using `.value_counts()`
   - Compare average feature values per class using `.groupby(60).mean()`

2. **Data Preparation**
   - Separate features (`X`) from labels (`Y`)
   - Split data into training and test sets (90% train / 10% test) using `train_test_split`, with stratification on the label to preserve class balance

3. **Model Training**
   - Train a `LogisticRegression` model from scikit-learn on the training data

4. **Model Evaluation**
   - Evaluate accuracy on both the training and test sets using `accuracy_score`

5. **Predictive System**
   - Accept a single new sonar reading (60 feature values)
   - Reshape it appropriately for prediction
   - Output whether the object is classified as a **Rock** or a **Mine**

## 🧠 Model

| Detail            | Value                    |
|--------------------|---------------------------|
| Algorithm          | Logistic Regression       |
| Library            | scikit-learn               |
| Task Type          | Binary Classification      |
| Input Features     | 60 (sonar frequency bands) |
| Output Classes     | `R` (Rock), `M` (Mine)     |

## 📈 Results

The model's performance is evaluated separately on the training and test sets to check for overfitting and confirm generalization:

```python
Accuracy on training data: <printed at runtime>
Accuracy on test data:     <printed at runtime>
```

*(Run the notebook to view the exact accuracy scores for your environment.)*

## 🔍 Example Prediction

```python
input_data = (0.0283, 0.0599, 0.0656, ..., 0.0045, 0.0079)  # 60 values

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 'R':
    print('The object is a Rock')
else:
    print('The object is a Mine')
```

## 🛠️ Tech Stack

- Python 3.11
- NumPy
- Pandas
- scikit-learn

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn
   ```
3. Place `sonar data.csv` in the project directory
4. Run the notebook:
   ```bash
   jupyter notebook Rock_vs_Mine.ipynb
   ```

## 📂 Project Structure

```
├── Rock_vs_Mine.ipynb     # Main notebook: data loading, training, evaluation, prediction
├── sonar data.csv         # Sonar dataset (features + labels)
└── README.md              # Project documentation
```

## 💡 Future Improvements

- Experiment with additional classifiers (Random Forest, SVM, Neural Networks) for comparison
- Add cross-validation for more robust accuracy estimates
- Build a simple web interface (e.g., Flask/Streamlit) to test predictions interactively
- Add feature scaling/normalization experiments to assess impact on performance

## 📄 License

This project is open-source and available for educational and personal use.
