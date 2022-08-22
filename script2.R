library(tidyverse)
library(tidymodels)
library(caret)
library(themis)

read_csv('train.csv') -> df
read_csv('test.csv') -> test

sum(is.na(df)) # Check for NA

df <- df |> 
  mutate(label = as.factor(label)) |> 
  select(-user_id)

# The next line of code asks R to check if 
# each vector or column in DF is <= 6. we store 
# The result of the check is stored in abc.

# abc <-sapply(df, n_distinct) <= 6
# abd <- sapply(test, n_distinct) <= 6

# We then ask R to convert every column which
# has distinct values <= 6 as factor.

# df<- df |> 
#   mutate_if(abc,as.factor) |> 
#   mutate(X13 = as.factor(X13),
#          X40 = as.factor(X40))

# test<- test |> 
#   mutate_if(abd, as.factor)


df |> 
  count(label)

prop.table(table(df$label))

# Partitioning data for local training and testing
set.seed(45)
samp <- createDataPartition(df$label, p = .85, list = FALSE)
tr <- df[samp,] 
ts <- df[-samp,]


#-----------------------------------

#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = tr) |>  
  step_impute_median(all_numeric_predictors()) |> 
  step_zv(all_predictors()) |>   # Removes variables containing single values
  step_nzv(all_predictors()) |>   #%>% # Removes variables that are sparse and unbalanced
  step_normalize(all_numeric_predictors()) |>   # Transforms data to have mean 0 and SD 1
  step_smote(label)

# juicing pre-processed dataframe
pTrain<- trRecipe |> 
  prep() |> 
  juice()

pTest<- trRecipe |> 
  prep() |> 
  bake(ts)

# Creating Data Matrix
trMatrix<- data.matrix(pTrain[,-33])
trY<- as.numeric(pTrain$label)-1

tsMatrix<- data.matrix(pTest[,-33])
tsY<- as.numeric(pTest$label)-1

dTrain<- xgb.DMatrix(data = trMatrix, label = trY)
dTest<- xgb.DMatrix(data = tsMatrix, label = tsY)

#-----------------------------------
library(xgboost)
set.seed(123)

# booster = 'gbtree': Possible to also have linear boosters as your weak learners.
params_booster <- list(booster = 'gbtree',
                       eta = 1, gamma = 0,
                       max.depth = 2,
                       subsample = 1,
                       colsample_bytree = 1,
                       min_child_weight = 1,
                       objective = "binary:logistic")

bst.cv <- xgb.cv(data = trMatrix, 
                 label = trY,
                 params = params_booster,
                 nrounds = 300, 
                 nfold = 5,
                 print_every_n = 20,
                 verbose = 2)


bstSparse <- xgboost(data = trMatrix, label = trY, nrounds = 320, params = params_booster)


test1<- trRecipe |> 
  prep() |> 
  bake(test)

testPredictDf<- data.matrix(test1)


pred<- predict(bstSparse, testPredictDf)
pred<- as.numeric(pred>.5)
submission<- as.data.frame(cbind(user_id = test$user_id, prediction = pred))


write.csv(submission, "paulie.csv")
