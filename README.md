# 🏡 Rent vs Buy Decision Assistant (India 🇮🇳 / UK 🇬🇧)

This project helps users make informed, data-driven decisions about whether to rent or buy a home — with support for country-specific logic, editable inputs, and even an optional ML-powered house price predictor.

Built with beginners in mind, this app combines core data science skills with an intuitive user interface using **Streamlit**.

---
## Try it out!
https://rushirentorbuy.streamlit.app/

---

## 🎯 Features

- **Country Toggle** – Choose India or the UK to auto-fill financial assumptions like interest rate, loan tenure, and property tax.
- **Editable Inputs** – Override rent, house price, down payment %, and more to match your real-life situation.
- **City Presets (Optional)** – Select a city to load local rent and property price estimates.
- **House Price Prediction (Optional ML)** – Enable a regression-based model that simulates appreciation and adjusts the cost of buying accordingly.
- **Visual Comparison** – Line chart showing Rent vs Buy cost over time.
- **Final Recommendation** – Based on total cost across years.

---

## 📊 Logic Behind the Scenes

- **EMI Calculation**  
  \[
  EMI = P \times \frac{r(1 + r)^n}{(1 + r)^n - 1}
  \]  
  Where `P` = loan principal, `r` = monthly interest rate, `n` = total months

- **Total Rent Cost** = Monthly Rent × 12 × Years  
- **Total Buy Cost** = EMI × 12 × Tenure + Property Tax × 12 × Years  
- *(If ML is enabled, appreciation is subtracted from Buy Cost)*

---

## 💡 Why This Project?

This isn’t just a calculator. It’s a compact, deployable data product that integrates:
- 📈 Real-world modeling
- 🧠 Basic ML integration
- 🛠️ Pythonic deployment
- 📊 Clear data storytelling

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
