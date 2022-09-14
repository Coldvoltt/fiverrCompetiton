library(tidyverse)
library(catboost)
library(caret)
library(tidymodels)
library(themis)
library(tidy.outliers)

read_csv('train.csv') -> df
read_csv('test.csv') -> testdf

df <- df |>
  mutate(label = as.factor(label)) |> 
  select(-user_id)


#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = df) |>  
  step_impute_median(all_numeric_predictors()) |> # Median Imputation
  step_zv(all_predictors()) |>   # Removes variables containing single values
  step_nzv(all_predictors()) |>  # Removes variables that are sparse and unbalanced
  step_normalize(all_numeric_predictors()) |>   # Transforms data to have mean 0 and SD 1
  step_smote(label)   # Tackle class imbalance

# juicing pre-processed dataframe
pTrain<- trRecipe |> 
  prep() |> 
  juice() |> 
  as.data.frame()

library(xgboost)

y = as.numeric(levels(pTrain$label))[pTrain$label]

xgbFinal<- xgboost(data=data.matrix(pTrain[,-33]),
             label= y,
             nround = 5000,
             objective = "binary:logistic",
             eval_metric = "logloss")

val<- testdf

valMat<-trRecipe |>
  prep() |>
  bake(val) |> 
  data.matrix()


pred<- predict(xgbFinal, newdata = valMat)
pred<- ifelse(pred>.5,1,0)


submission4<- as.data.frame(cbind(user_id = testdf$user_id, prediction = predDf))


write.csv(submission4, "xgboost4.csv")
