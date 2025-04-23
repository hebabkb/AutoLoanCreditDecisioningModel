

[Erdős Institute Data Science Boot Camp](https://github.com/TheErdosInstitute/data-science-spring-2025), Spring 2025.

- View my 5 mins [presentation](https://www.erdosinstitute.org/project-database/spring-2025/data-science-boot-camp/auto-loan-credit-decisioning-model)

<h1>Auto Loan Credit Decisioning Model</h1>

<h3>Team Members:</h3>

[Heba Bou KaedBey](https://github.com/hebabkb).

## Background:
An auto loan is a type of secured credit that allows consumers to borrow money to purchase a vehicle, with said vehicle used as collateral on the loan. Prospective borrowers may apply for an auto loan individually or jointly. Joint borrowers are typically spouses, or a child and a parent. Borrowers repay the loan in fixed installments over a set period, with interest charged on the outstanding balance amount. Defaulting on the loan could cause considerable damage to a person's credit score and impact their future creditworthiness.

## Setting:
Assuming that we are working in the consumer lending modeling team of a hypothetical financial institution and are assigned a task to enhance the current application decisioning process with a focus on providing equal credit lending opportunity to all applicants. We want to build a credit decisioning model based on the Auto Loan applicants' credit quality information. The model will aim to identify the applicants with good credit quality and unlikely to default.

## Information Provided:
We are given Auto Loan account data containing one binary response called 'bad_flag' in the datasets and a set of potential predictor variables from those Auto Loan accounts. Each record represents a unique account. There are two datasets: 1. Training data with around 21,000 records 2. Testing data with around 5,400 accounts.

## Objectives:
- Conducting an exploratory analysis to provide data summaries and necessary pre-processing for modeling.
- Developing and assessing machine learning models such as Logistic Regression, Random Forest, Gradient Boosting (XGBoost), Stacking Models.
- Comparing the results from these models to recommend the most effective approach for approving loan applications.
- Addressing critical business questions related to model transparency, gender bias, and potential racial discrimination.

## Dataset Description: 

- Number of Features: 42
- Target Variables: bad\_flag (highly imbalanced):


![](https://github.com/hebabkb/AutoLoanCreditDecisioningModel/blob/main/Presentation%20Images/download-1.png)

![](https://github.com/hebabkb/AutoLoanCreditDecisioningModel/blob/main/Presentation%20Images/download.png)

- Pre-Processing: Removing columns with more than 80% missing values, capping outliers, scaling features, encoding categorical features, removing correlated features with high correlation threshold.

## Model Overview:

### Models Explored: 

Logistic Regression: Base model
Random Forest
Balanced Random Forest
XGBoost
Stacking Ensemble (Random Forest and XGBoost as base learners and Logistic Regression as meta learner) (final choice)


### Performance Metrics:

Models were compared based on recall (class 1, bad loans), PR AUC and fairness metrics.

## Model Evaluation (Without applying sampling techniques):

### Chosen Model: 
Stacking Model (XGBoost and RF as base learners with LR as meta learner), achieved a fair performance.

![](https://github.com/hebabkb/AutoLoanCreditDecisioningModel/blob/main/Presentation%20Images/Screen Shot 2025-04-23 at 4.58.17 PM)

![](https://github.com/hebabkb/AutoLoanCreditDecisioningModel/blob/main/Presentation%20Images/download-5.png)

![](https://github.com/hebabkb/AutoLoanCreditDecisioningModel/blob/main/Presentation%20Images/download-3.png)

### Classification Results: 

PS: I applied SMOTE (oversampling technique) for class imbalance but it led to overfitting. 

## Interpretability of the Model (Using SHAP):

This SHAP plot ranks the most influential features in our model for predicting bad loans. For instance, low FICO scores and high LTV ratios significantly increase the predicted risk of default. These insights align with lending domain knowledge and provide transparency into how the model makes decisions. 

## Fairness Analysis by Gender and Race:

We evaluated the model’s fairness using group-based metrics across Gender and Race. We focused on:

- TPR (True Positive Rate) — % of bad loans correctly flagged

- FPR (False Positive Rate) — % of good loans incorrectly flagged

- Approval Rate — overall likelihood of a loan being approved

- EOD (Equal Opportunity Difference) — gap in TPRs across groups

- DI (Disparate Impact) — ratio of approval rates (ideal ≈ 1)

EOD = 0.0266 — very small difference in TPR between Female and Male.

DI = 0.9967 — approval rates are almost equal (ideal = 1).

Conclusion: The model treats Female and Male applicants equitably with minimal disparity.

EOD = 0.0776 — acceptable range; gap in TPRs has been reduced after threshold tuning.

DI = 0.8889 — within the legal “safe harbor” range (0.8 – 1.25).

Conclusion: The model shows improved racial fairness, though some differences remain, especially in approval rates between Black and Asian applicants.









