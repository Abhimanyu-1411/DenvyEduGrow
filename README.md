# DenvyEduGrow
DenvyEduGrow summer school 2025

## 📚 Student Dataset Preprocessing

This repository contains two Python scripts for cleaning, transforming, and preparing **student performance datasets**. The datasets include information like **Math** and **Science** scores, which are preprocessed through a series of steps to ensure they are analysis- and model-ready.

---

### 📁 Files Included

* `Set_11.py` – Preprocessing for `SET-11.csv`
* `Set_12.py` – Preprocessing for `SET-12.csv`

---

## ✅ Preprocessing Steps Performed

### 🔹 Common Tasks for Both Scripts

1. **Load CSV file** using `pandas`
2. **Handle missing values**

   * Replace missing values in numerical columns with the column mean
   * Fill non-numerical NaNs (where applicable) with `"Unknown"`
3. **Remove duplicate records** to ensure data consistency
4. **Normalize numerical columns** using **Min-Max Scaling** (range 0 to 1)
5. **Discretize** numeric scores using **equal-width binning**

   * Scores divided into 3 categories: `'Low'`, `'Medium'`, `'High'`
6. **Smooth noisy data** by:

   * Binning values and replacing them with **bin means**
   * Measuring **standard deviation before and after** smoothing to quantify noise reduction

---

## 🧠 Additional Notes

### 🔸 `Set_11.py` also includes:

* Discretization of **Science** scores into:

  * `Poor` (0–49), `Average` (50–69), `Good` (70–89), `Excellent` (90–100)
* Application of smoothing specifically on the **Math** column

### 🔸 `Set_12.py` extends functionality to:

* Automatically handle **all numeric columns**
* Dynamically compute and log **noise reduction** for each numeric feature
* Print summaries and transformation reports

---

## 📦 Requirements

Install dependencies using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 🚀 How to Run

```bash
# Run preprocessing script for Set 1
python Set_11.py

# Run preprocessing script for Set 2
python Set_12.py
```

---

## 📤 Output

Each script prints:

* Dataset shape before and after cleaning
* Missing values handled
* Normalized values
* Binning category counts
* Smoothed values and noise reduction stats
* Final summary with key columns

Optionally, you can save the final processed dataset using:

```python
df.to_csv("processed_output.csv", index=False)
```

---

## 📬 Contact

For questions, issues, or improvements, feel free to reach out at [abhimanyudebsocial@gmail.com) or open an issue in the repository.
