library(tidyverse)
library(tidymodels)
library(caret)
#library(themis)

read_csv('train.csv') -> df
read_csv('test.csv') -> test

sum(is.na(df)) # Check for NA
df<- na.omit(df) # Remove 6 NAs

# Because of numerous features, we may not plot correlation matrix..
# We may just look out for top 10 most correlated responses.
# Using the Lares Bernardo Larse library, we may achieve this.  

devtools::install_github("laresbernardo/lares")
library(lares)

corr_cross(train[-c(1,2),], # name of dataset
           max_pvalue = 0.05, # display only significant correlations (at 5% level)
           top = 10 # display top 10 couples of variables (by correlation coefficient)
)

# We may observe that variables x10 and x20 are highly correlated. 0.85
# We may observe that variables x4 and x5 are highly correlated. 0.78
# We may need to drop one of it later.

# Partitioning data for local training and testing
samp <- createDataPartition(df$label, p = .85, list = FALSE)
tr <- df[samp,] 
ts <- df[-samp,]


# Creating a model
#Define the model
gbm<- boost_tree() %>% 
  set_engine('C5.0') %>% 
  set_mode('classification') %>% 
  translate()
#-----------------------------------

#Creating a recipe with pre-processing
trRecipe <- recipe(label~., data = tr[,-2]) %>% 
  step_zv(all_predictors()) %>%  # Removes variables containing single values
  step_nzv(all_predictors()) %>%  #%>% # Removes variables that are sparse and unbalanced
  step_normalize(all_predictors())# Transforms data to have mean 0 and SD 1


# Checking effect of pre-processing on dataset
trx<- prep(trRecipe) %>% 
  juice()

# We may observe reduction in dimension
dim(trx)


# Create a workflow
gbm.wf<- workflow() %>% 
  add_model(gbm)

# Add Recipe to workflow
gbmRecipeWf<- add_recipe(gbm.wf, trRecipe)

#Fit a model
gbmMod<-  
  fit(gbmRecipeWf, data = tr[,-2])

gbmPred<- predict(gbmMod, new_data = ts)
gbmPred<- data.frame(celltype=gbmPred$.pred_class)

pred5<- cbind(location = ts$location, celltype = gbmPred)
