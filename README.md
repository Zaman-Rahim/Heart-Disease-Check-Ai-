# ❤️ HeartCheck AI

A machine learning web app that estimates the risk of heart disease from a patient's clinical details. Built with **Python**, **scikit-learn** and **Streamlit**.

This is my **first machine learning project**, built while learning ML: from data cleaning and model training all the way to a live deployed app.

### 🚀 [**Try the live app → heartdiseasecheckai.streamlit.app**](https://heartdiseasecheckai.streamlit.app)

> ⚠️ **Disclaimer:** This project is for **educational and demonstration purposes only**. It is not a medical device and must not be used as a substitute for professional medical diagnosis or advice.

---

## 📌 Project Summary

| | |
|---|---|
| **Problem** | Binary classification: predict whether a patient has heart disease (1) or not (0) |
| **Dataset** | `heart.csv`: 918 patients, 11 features + 1 target (`HeartDisease`) |
| **Model** | Logistic Regression |
| **Test accuracy** | **86.41%** (80/20 train-test split, `random_state=42`) |
| **Deployment** | Streamlit Community Cloud |

## 🔄 Workflow

### 1. Data exploration
- Loaded the dataset: **918 rows × 12 columns**
- Checked for missing values: **none found**
- Checked for duplicate rows: **none found**
- Plotted histograms (with KDE) of the numeric features and a correlation heatmap

### 2. Data cleaning
Some columns contained `0` values that are not physiologically valid, so they were treated as missing data:
- **`RestingBP`**: zeros replaced with the mean of the non-zero values
- **`Cholesterol`**: **172** zero values replaced with the mean of the non-zero values

### 3. Preprocessing
- **One-hot encoding** of categorical features with `pd.get_dummies(drop_first=True)`
- **Standardization** with `StandardScaler` on the numeric columns: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`
- **Train-test split:** 80% training, 20% testing (`random_state=42`)

### 4. Model training
- Trained a **Logistic Regression** model with scikit-learn
- Achieved **86.41% accuracy** on the test set

### 5. Deployment
- Saved the model, scaler and training columns with `joblib`
- Built the interface with **Streamlit** and deployed it on **Streamlit Community Cloud**

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | M / F |
| `ChestPainType` | TA (Typical Angina), ATA (Atypical Angina), NAP (Non-Anginal Pain), ASY (Asymptomatic) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `RestingECG` | Normal, ST (ST-T wave abnormality), LVH (left ventricular hypertrophy) |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak exercise ST segment: Up, Flat, Down |
| `HeartDisease` | **Target**: 1 = heart disease, 0 = normal |

## ✨ App Features

- 🎨 Colorful, modern interface with a light theme
- 🧾 Simple inputs (sliders, dropdowns, radio buttons), with no technical encoding needed
- 🧮 Instant prediction with a risk probability bar
- ☁️ Publicly accessible through a single link

## 🛠️ Tech Stack

- **Python**
- **pandas** and **NumPy** for data handling
- **Matplotlib** and **Seaborn** for visualization
- **scikit-learn** for preprocessing, modeling and evaluation
- **joblib** for saving the model
- **Streamlit** for the web interface

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml        # Light theme configuration
├── app.py                 # Streamlit application
├── heart_model.pkl        # Trained Logistic Regression model
├── scaler.pkl             # Fitted StandardScaler
├── columns.pkl            # Training column order
└── requirements.txt       # Dependencies
```

## ▶️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Zaman-Rahim/Heart-Disease-Check-Ai-.git
cd Heart-Disease-Check-Ai-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the app
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## 📝 Limitations and Next Steps

This is a learning project, and there is room to improve it:

- Only **accuracy** is reported so far. For a medical problem, **recall** and **precision** matter too, because missing a patient is costly.
- The scaler was fitted on the full dataset before the train-test split. Fitting it on the training set only would avoid data leakage and give a more honest accuracy.
- Only one model (Logistic Regression) was tried. Comparing it with Random Forest, SVM or XGBoost, and adding cross-validation, would be a good next step.
- Zero values in `Cholesterol` and `RestingBP` were filled with the mean. Median-based or model-based imputation could be explored.
- The dataset is small (918 patients), so results may not generalize to real-world populations.

## 👤 Author

**Rahim**, Computer Science and Engineering student
GitHub: [@Zaman-Rahim](https://github.com/Zaman-Rahim)

---

⭐ If you found this project useful, consider giving it a star!
