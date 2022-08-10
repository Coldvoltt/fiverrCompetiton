library(tidyverse)
library(tidymodels)
library(caret)
#library(themis)

read_csv('train.csv') -> df
read_csv('test.csv') -> test

sum(is.na(df)) # Check for NA
df<- na.omit(df) # Remove 6 NAs

# The next line of code asks R to check if 
# each vector or column in DF is <= 6. we store 
# The result of the check is stored in abc.

abc <-sapply(df, n_distinct) <= 6
abd <- sapply(test, n_distinct) <= 6

# We then ask R to convert every column which
# has distinct values <= 6 as factor.

df<- df |> 
  mutate_if(abc,as.factor) |> 
  mutate(X13 = as.factor(X13),
         X40 = as.factor(X40))

test<- test |> 
  mutate_if(abd, as.factor)
# Because of numerous features, we may not plot correlation matrix..
# We may just look out for top 10 most correlated responses.
# Using the Lares Bernardo Larse library, we may achieve this.  

# devtools::install_github("laresbernardo/lares")
library(lares)

corr_cross(df[-c(1,2),], # name of dataset
           max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 10 # display top 10 couples of variables (by correlation coefficient)
)

# # We convert label to factor.
# df<- df |> 
#   mutate(label = as.factor(label))

# We also need to convert some features that are boolean [0,1] in nature to factor for
# the model to better understand the data structure

# We may observe that variables x10 and x20 are highly correlated. 0.85
# We may observe that variables x4 and x5 are highly correlated. 0.78
# We may need to drop one of it later.

# Partitioning data for local training and testing
set.seed(45)
samp <- createDataPartition(df$label, p = .90, list = FALSE)
tr <- df[samp,] 
ts <- df[-samp,]


# Creating a model
# Define the models

# Gradient Boosting Model
gbm<- boost_tree() |> 
  set_engine('C5.0')  |>  
  set_mode('classification')  |>  
  translate()
  
#-----------------------------------

#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = tr[,-2]) %>% 
  step_zv(all_predictors()) %>%  # Removes variables containing single values
  step_nzv(all_predictors()) %>%  #%>% # Removes variables that are sparse and unbalanced
  step_normalize(all_numeric_predictors())  # Transforms data to have mean 0 and SD 1

# Checking effect of pre-processing on dataset
trx<- trRecipe |> 
  prep() |> 
  juice()

# We may observe reduction in dimension
dim(trx)

# ----------------------------------

# Create a workflow for GBM
gbm.wf<- workflow()  |>  
  add_model(gbm) 

# Add Recipe to GBM workflow
gbmRecipeWf<- add_recipe(gbm.wf, trRecipe)

#Fit GBM model
gbmMod<-  
  fit(gbmRecipeWf, data = tr[,-2]);Sys.time()

gbmPred<- predict(gbmMod, new_data = ts)
gbmPred<- data.frame(label=gbmPred$.pred_class)

confusionMatrix(gbmPred$label, ts$label, 
                mode = "everything")


label<- predict(gbmMod, new_data = test)
submit<- data.frame(cbind(user_id = test$user_id, prediction =label))
write.csv(submit, "AlexanderPaul.csv")


# -------------------------------------
library(MLmetrics)
F1_Score(gbmPred$label,ts$label)
# -------------------------------------

