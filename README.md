# Customer Churn Prediction (Telecom)

## Business Problem
Customer churn (when a customer cancels their subscription) is a massive revenue drain for telecom companies. It costs 5x more to acquire a new customer than to retain an existing one. 

The goal of this project is to build a Machine Learning model that predicts which customers are at risk of churning, allowing the business to deploy targeted retention strategies.

## Tech Stack
- Python, Pandas, NumPy
- Matplotlib & Seaborn (EDA)
- Scikit-Learn (Preprocessing, Baseline Modeling, Evaluation)
- XGBoost (Advanced Modeling)
- Jupyter Notebooks

## Key Insights
1. **Contract Type:** Customers on month-to-month contracts are the most likely to churn. Long-term contracts (1 or 2 years) heavily reduce churn.
2. **Fiber Optic:** Customers using Fiber Optic internet have a surprisingly high churn rate, indicating potential pricing or quality-of-service issues.
3. **Tenure:** The first year is critical. If a customer stays past their first year, their likelihood of churning drops significantly.

## Model Performance
Because the dataset was highly imbalanced (73% stayed, 27% churned), accuracy alone was not a reliable metric. 

- **Baseline (Logistic Regression):** Achieved a 72.6% accuracy, but struggled with feature scaling.
- **Advanced (XGBoost):** Achieved the best overall performance (73.0% accuracy) while successfully catching the highest number of actual churners (Recall = 0.80), reducing False Negatives. I used `scale_pos_weight` to force the model to pay attention to the minority class.

## How to Run
1. Clone this repository.
2. Set up a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   pip install -r requirements.txt