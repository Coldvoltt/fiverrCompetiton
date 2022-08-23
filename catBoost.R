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



trY<- as.numeric(pTrain$label)
tsY<- as.numeric(pTest$label)

train<- catboost.load_pool(data = pTrain[,-1], label = trY)
test<- catboost.load_pool(data = pTest[,-1], label = tsY)

params<- list(iterations = 1500,
              use_best_model = TRUE)

catBoostMod<- catboost.train(train,test, params)



val<- testdf |> 
  select(-user_id)

# vald<-trRecipe |> 
#   prep() |> 
#   bake(val) 

vald<- val|> 
  catboost.load_pool()


predDf<- catboost.predict(catBoostMod,vald)
predDf<- as.numeric(predDf>.5)
submission2<- as.data.frame(cbind(user_id = testdf$user_id, prediction = predDf))


write.csv(submission2, "paulie2.csv")
