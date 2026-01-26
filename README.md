
# House Price Prediction Project

A complete machine learning project to predict house sale prices using a standard real-estate dataset.

Built with scikit-learn, featuring data cleaning, preprocessing pipeline, handling missing values, categorical encoding, log transformation, and model evaluation.

## Project Goals

- Build a reliable house price prediction model
- Learn and apply end-to-end ML workflow
- Handle real-world data issues (missing values, rare categories, skewed target)
- Achieve strong performance on validation data

## Final Model Performance (Validation Set)

- **R² Score**: ≈ 0.89  
- **RMSE**: ≈ $27,000  
- **Estimated RMSLE** (log-scale error): ≈ 0.13 – 0.16 range  
  → Solid mid-intermediate level result for a single model

## Key Features & Techniques Used

- **Data Preprocessing Pipeline** using `ColumnTransformer` + `Pipeline`
- Numerical features: imputation (median), scaling
- Categorical features: imputation ('None' for missing), OneHotEncoding with `handle_unknown='ignore'`
- Target transformation: `np.log1p` on SalePrice → reverse with `np.expm1`
- Handled common issues:
  - Missing columns / wrong column names
  - NaN values crashing models
  - Unseen categories in test set
  - Infinity values in predictions
- Model: [RandomForestRegressor / HistGradientBoostingRegressor / XGBoost – choose what you used]

## Project Structure

```
House-Price-Prediction/
├── train.csv                   # Training data
├── test.csv                    # Test data (no target column)
├── submission.csv              # Latest prediction file
├── House_Price_Prediction.ipynb  # Main Jupyter notebook
├── README.md
└── .venv/                      # Virtual environment (not uploaded)
```

## How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/VishnuVardhanReddyKothapuli/House_price_prediction
   cd House-Price-Prediction
   ```

2. Install dependencies (recommended: use virtual environment)
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   (Create `requirements.txt` if you don't have one yet:)
   ```
   pandas
   numpy
   scikit-learn
   # optional but recommended:
   xgboost
   lightgbm
   matplotlib
   seaborn
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook House_Price_Prediction.ipynb
   ```

4. To generate new predictions:
   - Run all cells up to model fitting
   - Run the test prediction cells at the bottom

## Results & Learnings

- Started with many common beginner errors (KeyError, NaN in models, unseen categories, inf predictions)
- Successfully built a full sklearn pipeline
- Reached validation performance better than many starter notebooks
- Ready for further improvements: better models, feature engineering, cross-validation, ensembling

## Future Improvements (planned / ideas)

- Try XGBoost / LightGBM / CatBoost
- Add feature engineering (total living area, bathroom count, house age, interactions)
- Use target encoding for high-cardinality features
- Cross-validation & hyperparameter tuning
- Ensemble multiple models
