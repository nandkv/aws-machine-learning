# Problem Statement
Identifying and catching fraudulent transactions.

# ETL Strategy:
## 
Split the dataset in to 3 datasets:
1) Training Dataset - 70 % of the dataset goes in to training the model
2) Validation Dataset - 15%  of the dataset to optimize the model 
3) Test Dataset - 15% of the dataset to test the model 

# Modelling Strategy:
##

Use the XGBoost Algorithm 

It is very fast and utilizes gradient boosting which is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
# Deployment Strategy

