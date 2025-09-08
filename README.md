# Prediction Model for Obesity Classification

This project develops and evaluates predictive models for **obesity classification** using a dataset of 2,372 samples that integrates demographic, anthropometric, clinical, lifestyle, and family history information.  

## Overview
Obesity is influenced by biological, behavioral, socioeconomic, and environmental factors.  
The goal of this project is to identify predictors of obesity beyond traditional biomarkers and to build interpretable, high-performing models for classification.

## Data
- **Samples:** 2,372  
- **Predictors:** 38 features  
  - Demographics  
  - Anthropometrics  
  - Biomarkers  
  - Lifestyle factors  
  - Family history  
- **Outcome:** Binary obesity status (prevalence ~20%)  

## Methods
- **Preprocessing**
  - Standardization of continuous predictors
  - Dummy encoding for categorical variables
  - Multicollinearity checks (VIF) and residual diagnostics
- **Models**
  - Logistic Regression (baseline)
  - Lasso Logistic Regression (L1 penalty for feature selection)
  - Random Forest (nonlinear interactions + feature importance)
- **Evaluation**
  - ROC curve & AUC metric

## Results
- Logistic Regression (baseline): **ROC-AUC = 0.743**  
- Lasso Logistic Regression: **ROC-AUC = 0.884** (best, 18 predictors)  
- Random Forest: **ROC-AUC = 0.803** (peaked with ~20 top features)

## Key Insights
- Regularization and feature selection improved both **accuracy** and **interpretability**.  
- Lasso regression provided the best trade-off between performance and simplicity.  
- A smaller, influential subset of predictors can inform **cost-effective screening tools** for obesity prevention and intervention.

---
