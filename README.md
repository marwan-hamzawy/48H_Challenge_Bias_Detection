# 48H_Challenge_Bias_Detection

# 🧠 Bias Detection & Explainability in AI Hiring Models

⚠️⚠️ **Important Note on Dataset Format vs Challenge Context** ⚠️⚠️
-- The dataset provided with the challenge was fully numerical and did not include any text fields (e.g., resume summaries or cover letters) as described in the challenge context. Therefore, we applied suitable tabular modeling techniques (Random Forest and XGBoost) and conducted fairness and explainability analysis accordingly, within the limits of the data format.

## 🚀 Challenge Overview
This project was built for the 48-hour challenge:  
**“Uncovering Bias and Explaining Decisions in a Text-Based Job Screening Model”**.

The task: simulate and audit a model that decides whether to hire applicants, and analyze if any **bias exists** (e.g., gender-based). The model is explainable via SHAP visualizations, and fairness metrics are reported.

---

## 📦 Dataset Description
The dataset contains structured, numerical hiring-related features such as:

- `Age`, `ExperienceYears`, `SkillScore`, `PersonalityScore`, etc.
- `RecruitmentStrategy`: one-hot encoded
- `Gender`: Binary (`0 = Female`, `1 = Male`) — **used as the sensitive attribute**

---

## ⚖️ Handling Data Imbalance

- The dataset showed **class imbalance** between `Hire` and `Not Hire` classes.
- We applied:
  - `class_weight='balanced'` in RandomForest
  - `scale_pos_weight` in XGBoost based on training set ratios

---

## 🧠 Model Used

Two classifiers were trained and evaluated:

1. **RandomForestClassifier** (with class weight)
2. **XGBoostClassifier** (with scale_pos_weight)

Features were scaled (except for encoded/binary columns), and models were trained with **stratified split**.

---

## 📊 Model Performance

On the test set of 300 samples:

### 🔹 Random Forest
Confusion Matrix:
[[204 3]
[ 23 70]]

Accuracy: 91.3%
Macro F1 Score: 0.892
ROC AUC Score: 0.869

### 🔹 XGBoost
Confusion Matrix:
[[202 5]
[ 17 76]]

Accuracy: 92.7%
Macro F1 Score: 0.911
ROC AUC Score: 0.897 

---

## 🧪 Fairness Analysis

The following **group fairness metrics** were evaluated by `Gender`:

| Metric             | Female (0) | Male (1)  |
|--------------------|------------|-----------|
| Selection Rate     | 22.2%      | 31.4%     |
| True Positive Rate | 75.0%      | 86.8%     |
| False Positive Rate| 1.9%       | 2.9%      |

📉 **Average Odds Difference** = `0.0639` → Moderate disparity  
> AOD combines both FPR and TPR disparity between gender groups.

---

## 🧠 Explainability (SHAP)

We used **SHAP TreeExplainer** on the XGBoost model to visualize:

- 📊 Global Feature Importance
- 🔍 Detailed dot summary plots for 5 predictions (3 Hire, 2 No-Hire)

Key findings:

- Features like `SkillScore`, `InterviewScore`, and `PersonalityScore` have high predictive weight.
- `Gender` had **low global SHAP impact**, but fairness metrics showed **outcome disparities**, suggesting potential **proxy bias** through other features.


