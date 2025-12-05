# API.R

#* @apiTitle Diabetes Prediction API
#* @apiDescription Predict diabetes and inspect model performance.

library(plumber)
library(tidymodels)
library(dplyr)
library(ggplot2)
library(yardstick)

tidymodels_prefer()


# ------------------------------------------------------------
# 1. Read and prepare the data
# ------------------------------------------------------------

diabetes <- read.csv("./data/diabetes_binary_health_indicators_BRFSS2015.csv")

# Drop any index-like column if present
id_like_cols <- c("X", "...1")
keep_cols <- setdiff(names(diabetes), id_like_cols)
diabetes <- diabetes[, keep_cols, drop = FALSE]

diabetes2 <- diabetes |>
  mutate(
    Diabetes_binary      = factor(Diabetes_binary),
    HighBP               = factor(HighBP),
    HighChol             = factor(HighChol),
    CholCheck            = factor(CholCheck),
    Smoker               = factor(Smoker),
    Stroke               = factor(Stroke),
    HeartDiseaseorAttack = factor(HeartDiseaseorAttack),
    PhysActivity         = factor(PhysActivity),
    Fruits               = factor(Fruits),
    Veggies              = factor(Veggies),
    HvyAlcoholConsump    = factor(HvyAlcoholConsump),
    AnyHealthcare        = factor(AnyHealthcare),
    NoDocbcCost          = factor(NoDocbcCost),
    GenHlth              = factor(GenHlth),
    DiffWalk             = factor(DiffWalk),
    Sex                  = factor(Sex),
    Age                  = factor(Age),
    Education            = factor(Education),
    Income               = factor(Income)
  ) 


# List of my selected predictor names
my_predictors <- diabetes2 |>
  select(
    Diabetes_binary,
  HighBP,
  BMI,
  PhysActivity,
  Age,
  GenHlth,
  Smoker,
  Education
)

predictor_names <- setdiff(names(my_predictors), "Diabetes_binary")

# ------------------------------------------------------------
# 2. Recipe
# ------------------------------------------------------------

diabetes_recipe <- recipe(Diabetes_binary ~ HighBP + BMI + PhysActivity + Age + 
                            GenHlth + Smoker + Education, data = diabetes2) %>%
  step_zv(all_predictors())

# ------------------------------------------------------------
# 3. Final model (Class)
# ------------------------------------------------------------

best_tree_spec <- decision_tree(
  cost_complexity = 1e-05,  # tuned value
  tree_depth      = 8,      # tuned depth
  min_n           = 5      
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


final_wf <- workflow() %>%
  add_model(best_tree_spec) %>%
  add_recipe(diabetes_recipe) %>%
  fit(data = diabetes2)

# ------------------------------------------------------------
# 4. Defaults for predictors
# ------------------------------------------------------------

numeric_defaults <- diabetes2 %>%
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE))) %>%
  as.list()

factor_defaults <- diabetes2 %>%
  summarise(across(where(is.factor), ~ names(which.max(table(.x))))) %>%
  slice(1) %>%
  as.list()

get_num_default <- function(nm, fallback = 0) {
  if (nm %in% names(numeric_defaults)) numeric_defaults[[nm]] else fallback
}

val_or_default <- function(x, default) {
  if (is.null(x) || x == "") default else x
}

# ------------------------------------------------------------
# 5. Pre-compute predictions for confusion matrix
# ------------------------------------------------------------

full_preds <- predict(final_wf, diabetes2, type = "class") %>%
  bind_cols(predict(final_wf, diabetes2, type = "prob")) %>%
  bind_cols(diabetes2 %>% select(Diabetes_binary))

# ------------------------------------------------------------
# 6. /pred endpoint
# ------------------------------------------------------------

#* Predict probability of diabetes
#*
#* Query parameters:
#*   HighBP, BMI, PhysActivity, Smoker, Age, Sex, Income, Education
#*   
#*37619
#* @get /pred
#*```` 
#*Example Functions:
#*
#*http://127.0.0.1:8000/pred?HighBP=1&BMI=22&Age=9
#*http://127.0.0.1:8000/pred?HighBP=1&PhysActivity=0&Smoker=1&GenHlth=5
#*http://127.0.0.1:8000/pred?Education=6&PhysActivity=1&Smoker=0
#*````
function(req, res) {
  
  p <- req$args  # list of query parameters as characters
  new_data <- tibble(
    HighBP = factor(
      val_or_default(p$HighBP, factor_defaults$HighBP),
      levels = levels(diabetes2$HighBP)
    ),
    BMI = as.numeric(val_or_default(p$BMI, get_num_default("BMI"))),
    Smoker = factor(
      val_or_default(p$Smoker, factor_defaults$Smoker),
      levels = levels(diabetes2$Smoker)
    ),
    PhysActivity = factor(
      val_or_default(p$PhysActivity, factor_defaults$PhysActivity),
      levels = levels(diabetes2$PhysActivity)
    ),
    GenHlth = factor(
      val_or_default(p$GenHlth, factor_defaults$GenHlth),
      levels = levels(diabetes2$GenHlth)
    ),
    Age = factor(
      val_or_default(p$Age, factor_defaults$Age),
      levels = levels(diabetes2$Age)
    ),
    Education = factor(
      val_or_default(p$Education, factor_defaults$Education),
      levels = levels(diabetes2$Education)
    )
    
    )
    
  
  
  # Align column order with training predictors
  new_data <- new_data %>% dplyr::select(all_of(predictor_names))
  
  prob <- predict(final_wf, new_data = new_data, type = "prob")
  
  list(
    input = new_data,
    prob_diabetes_1 = prob$.pred_1[1]
  )
  
}


# ------------------------------------------------------------
# 7. /info endpoint
# ------------------------------------------------------------

#* API info
#* @get /info
function() {
  list(
    name  = "Beza Wodajo",
    github_pages = "https://bezahub7.github.io/Final-Project/"  # git hub url
  )
}

# ------------------------------------------------------------
# 8. /confusion endpoint
# ------------------------------------------------------------

#* Confusion matrix plot for the fitted model
#* @serializer png
#* @get /confusion
function() {
  cm <- conf_mat(full_preds, truth = Diabetes_binary, estimate = .pred_class)
  p <- autoplot(cm, type="heatmap")          # create ggplot object
  print(p)                   # DRAW it to the device for plumber
  invisible(NULL)            # returns just image
}