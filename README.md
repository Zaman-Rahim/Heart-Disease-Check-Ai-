# ❤️ HeartCheck AI

A machine learning web app that estimates the risk of heart disease from a patient's clinical details. Built with **Python**, **scikit-learn** and **Streamlit**.

### 🚀 [**Try the live app → heartdiseasecheckai.streamlit.app**](https://heartdiseasecheckai.streamlit.app)

> ⚠️ **Disclaimer:** This project is for **educational and demonstration purposes only**. It is not a medical device and must not be used as a substitute for professional medical diagnosis or advice.

---

## ✨ Features

- 🎨 Clean, colorful and modern interface
- 🧮 Instant prediction with a risk probability bar
- 🧾 Simple inputs: sliders, dropdowns and radio buttons, with no technical encoding needed
- ☁️ Deployed on Streamlit Community Cloud, so anyone with the link can use it

## 🧠 How it works

1. The user enters 11 clinical details in the web form.
2. The app one-hot encodes the categorical values and aligns them with the training columns.
3. Numeric features are scaled with the same `StandardScaler` used during training.
4. The trained model predicts whether the risk is higher or lower, along with a probability.

## 📊 Dataset

The model is trained on the **Heart Failure Prediction** dataset with 11 features and a `HeartDisease` target.

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | M / F |
| `ChestPainType` | TA, ATA, NAP, ASY |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dl) |
| `FastingBS` | Fasting blood sugar > 120 mg/dl (1 = yes, 0 = no) |
| `RestingECG` | Normal, ST, LVH |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y / N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Up, Flat, Down |

## 🛠️ Tech Stack

- **Python**
- **pandas** for data handling
- **scikit-learn** for modeling and scaling
- **joblib** for saving the model
- **Streamlit** for the web interface

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml        # Light theme configuration
├── app.py                 # Streamlit application
├── heart_model.pkl        # Trained model
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

## 🔮 Future Improvements

- Add model explanation (feature importance / SHAP)
- Compare multiple models inside the app
- Add a Bengali language toggle
- Show recall and precision alongside accuracy

## 👤 Author

**Rahim**, Computer Science and Engineering student
GitHub: [@Zaman-Rahim](https://github.com/Zaman-Rahim)

---

⭐ If you found this project useful, consider giving it a star!
