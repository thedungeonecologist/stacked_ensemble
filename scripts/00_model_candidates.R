#' ---
#' title: "Read Data"
#' author: "Darren Shoemaker"
#' ---

# Libraries

library(here)
library(tidyverse)
library(themis)
library(baguette)
library(bench)
library(future)
library(tidymodels)

tidymodels_prefer()

#[1] Read Data ----

dat <- read.csv(file = here('data/raw', 'svm_synthetic.csv')) %>% 
  mutate(across(.cols = 1:45, \(x) factor(x, order = T, levels = c(0, 1, 2, 3, 4, 5))))

impairment_metrics <- colnames(dat[1:45])

temp_dat <- dat %>% 
  select(1,46:69)

# Split Test-Train Datasets

set.seed(42069)

train <- rsample::initial_split(temp_dat, strata = SHALLOW, prop = 0.8)
train_set <- training(train)
test_set <- testing(train)

train_folds <- vfold_cv(train_set, strata = SHALLOW, repeats = 5)

# Normalized and Polynomial Interaction Recipes

normalized_rec <- recipe(SHALLOW ~ ., data = train_set) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_upsample(all_outcomes())

# poly_rec <- normalized_rec %>% 
#   step_poly(all_predictors()) %>% 
#   step_interact(~ all_predictors():all_predictors()) %>% 
#   step_upsample(all_outcomes())

#[2] Model Specifications ----

## Logistic Regression

log_reg_spec <- multinom_reg() %>% 
  set_engine('glmnet') %>% 
  set_mode('classification')

## SVM RBF

svm_r_spec <- svm_rbf(cost = tune(),
                      rbf_sigma = tune()) %>% 
  set_engine('kernlab',
             scaled = T,
             prob.model = T) %>% 
  set_mode('classification')

## SVM Poly

svm_p_spec <- svm_poly(cost = tune(),
                       degree = tune(),
                       scale_factor = tune()) %>%
  set_engine('kernlab',
             scaled = T,
             prob.model = T) %>% 
  set_mode('classification')

## Random Forest (non-ordinal)

rf_spec <- rand_forest(mtry = tune(), 
                       min_n = tune(),
                       trees = tune()) %>% 
  set_engine('ranger') %>% 
  set_mode('classification')

## XGBoost

xgb_spec <- boost_tree(tree_depth = tune(),
                       learn_rate = tune(),
                       loss_reduction = tune(),
                       min_n = tune(),
                       sample_size = tune(),
                       trees = tune()) %>% 
  set_engine('xgboost',
             scaled = T,
             prob.model = T) %>% 
  set_mode('classification')

## Multilayer Perceptron Neural Network

mlp_spec <- mlp(hidden_units = tune(),
                penalty = tune(),
                epochs = tune()) %>% 
  set_engine('nnet',
             scaled = T,
             prob.model = T) %>% 
  set_mode('classification')

## Bagged Neural Network

bag_spec <- bag_mlp(hidden_units = tune(),
                    penalty = tune(),
                    epochs = tune()) %>% 
  set_engine('nnet',
             scaled = T,
             prob.model = T) %>% 
  set_mode('classification')

#[3] Workflow Sets ----

all_workflows <- workflow_set(
  preproc = list(normalized = normalized_rec),
  models = list(SVM_radial = svm_r_spec, 
                SVM_poly = svm_p_spec, 
                bag_nn = bag_spec, 
                mlp_nn = mlp_spec,
                boosting = xgb_spec,
                log_reg = log_reg_spec,
                RF = rf_spec)
)

#[4] Tuning and Evaluating ----

grid_ctrl <- control_grid(save_pred = T, parallel_over = 'everything', save_workflow = T)

future::plan(multisession)

bench::mark(

grid_results <- all_workflows %>% 
  workflow_map(seed = 42069,
               resamples = train_folds,
               grid = 25,
               control = grid_ctrl)
)

future::plan(sequential)