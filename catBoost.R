library(tidyverse)
library(catboost)
library(caret)
library(tidymodels)
library(themis)

read_csv('train.csv') -> df
read_csv('test.csv') -> testdf

df <- df |>
  mutate(label = as.factor(label)) |> 
  select(-user_id)


# Partitioning data for local training and testing
set.seed(45)
samp <- createDataPartition(df$label, p = .85, list = FALSE)
tr <- df[samp,] 
ts <- df[-samp,]

#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = tr) |>  
  step_impute_median(all_numeric_predictors()) |> # Median Imputation
  step_zv(all_predictors()) |>   # Removes variables containing single values
  step_nzv(all_predictors()) |>  # Removes variables that are sparse and unbalanced
  step_normalize(all_numeric_predictors()) |>   # Transforms data to have mean 0 and SD 1
  step_smote(label)   # Tackle class imbalance

# juicing pre-processed dataframe
pTrain<- trRecipe |> 
  prep() |> 
  juice()

pTest<- trRecipe |> 
  prep() |> 
  bake(ts)

trY=as.numeric(levels(pTrain$label))[pTrain$label]
tsY=as.numeric(levels(pTest$label))[pTest$label]

train_pool<- catboost.load_pool(pTrain[,-33], label =trY)
test_pool<- catboost.load_pool(pTest[,-33], label = tsY)

fit_params<- list(iterations = 105,
              random_seed = 50,
              loss_function = 'Logloss',
              use_best_model = TRUE)

catBoostMod<- catboost.train(learn_pool = train_pool, test = test_pool, params = fit_params)



val<- testdf

vald<-trRecipe |>
  prep() |>
  bake(val)

vald_pool<- vald|> 
  catboost.load_pool()


predDf<- catboost.predict(catBoostMod,vald_pool, prediction_type = "Class")

submission2<- as.data.frame(cbind(user_id = testdf$user_id, prediction = predDf))


write.csv(submission2, "catBoost.csv")
