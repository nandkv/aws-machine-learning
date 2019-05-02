# Problem Statement
Identifying and catching fraudulent bank transactions.
![Problem](https://s3.amazonaws.com/nand-aws-ml/one.png)


# ETL Strategy:
Split the dataset in to 3 datasets:
1) Training Dataset - 70 % of the dataset goes in to training the model
2) Validation Dataset - 15%  of the dataset to optimize the model 
3) Test Dataset - 15% of the dataset to test the model 
![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/two.png)
![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/three.png)
![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/six.png)


# Modelling Strategy:
*Use the XGBoost Algorithm*
￼![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/five.png)


It is very fast and utilizes gradient boosting which is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
￼![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/eight.png)
￼![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/seven.png)
![ETL Strategy](https://s3.amazonaws.com/nand-aws-ml/four.png)

# Deployment Strategy
Amazon SageMaker real-time hosted endpoint deployed on EC2 for flagging fraudelent transactions.
