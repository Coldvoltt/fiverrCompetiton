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


# ----------------------------------
# The model
rf<- rand_forest(trees = 100) |> 
  set_engine("ranger", num.threads = 3, seed = 123) |> 
  set_mode("classification")



#-----------------------------------

#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = tr) |>  
  step_impute_median(all_numeric_predictors()) |> 
  step_zv(all_predictors()) |>   # Removes variables containing single values
  step_nzv(all_predictors()) |>   #%>% # Removes variables that are sparse and unbalanced
  step_normalize(all_numeric_predictors()) |>   # Transforms data to have mean 0 and SD 1
  step_smote(label)

# Checking effect of pre-processing on dataset
trx<- trRecipe |> 
  prep() |> 
  juice()


rf_wf<- workflow() |> 
  add_model(rf) |> 
  add_recipe(trRecipe)


# ----------------------------------
# modeling the data
model_rf <- rf_wf |> 
  fit(data = tr)


rfPred<- predict(model_rf, new_data = ts)
rfPred<- data.frame(label=rfPred$.pred_class)

confusionMatrix(rfPred$label, ts$label, 
                mode = "everything")


label<- predict(model_rf, new_data = test)
submit<- data.frame(cbind(user_id = test$user_id, prediction =label))
write.csv(submit, "AlexanderPaul2.csv")


# -------------------------------------
library(MLmetrics)
F1_Score(gbmPred$label,ts$label)
# -------------------------------------

