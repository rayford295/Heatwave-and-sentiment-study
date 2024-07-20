# R script to run INLA models of increasing complexity
# WARNING: the script may take over a day to run

# Step 0: load packages and pre-processed data 
# Step 1: formulate a baseline model including spatiotemporal random effects and test different combinations of DLNM variables
# Step 0: load packages and pre-processed data
source("00_load_packages_data.R")

# run models of increasing complexity in INLA

# Step 1: fit a baseline model including spatiotemporal random effects

## formulate a base model including: 
# rw1: province-specific random effects to account for day-of-week variation (random walk cyclic prior) - with Random walk model of order 1 (RW1)
# https://inla.r-inla-download.org/r-inla.org/doc/latent/rw1.pdf
# bym2: week-specific spatial random effects to account for inter-week variation in spatial overdisperson and dependency structures (modified Besag-York-Mollie prior bym2)
# https://inla.r-inla-download.org/r-inla.org/doc/latent/bym2.pdf


## baseline model

# 
#baseformula2 <- Y ~ 1 + f(S1, model = "bym2", replicate = T2, graph = "texas/map.graph",
#     scale.model = TRUE, hyper = precision.prior2) 

# baseformula3 <- Y ~ f(T1, replicate = S1, model = "rw1", cyclic = TRUE, constr = TRUE,
#                       scale.model = TRUE,  hyper = precision.prior2) +
#   f(S1, model = "bym2", replicate = T2, graph = "cal/map.graph",
#     scale.model = TRUE, hyper = precision.prior2)  

baseformula5 <- Y ~ 1 + f(T1, replicate = S1, model = "rw1", cyclic = TRUE, constr = TRUE,
          scale.model = TRUE,  hyper = precision.prior2) +
  f(S1, model = "bym2", replicate = T2, graph = "alabama/map.graph",
    scale.model = TRUE, hyper = precision.prior2) +Vh+Vv

# formulas <- list(baseformula3,baseformula4)
# lab <- c("base3","base4")
# 
# 
# models <- lapply(1:length(formulas),
#                  function(i) {
#                    model <- mymodel(formula = formulas[[i]], data = data)
#                    save(model, file = paste0("South Carolina/2021/", lab[i],".RData"))})
# 
# # create table to store DIC and select best model
# table0 <- data.table(Model  =  c("base3","base4"),
#                      DIC = NA,
#                      logscore = NA)
# 
# for(i in 1:length(formulas))
# {
#   load(paste0("South Carolina/2021/",lab[i],".RData"))
#   table0$DIC[i] <- round(model$dic$dic, 2)
#   table0$logscore[i] <- round(-mean(log(model$cpo$cpo), na.rm = T), 3)
# }
# 
# # view table
# table0

# define position of best fitting model
# best.fit <- which.min(table0$DIC)
# 
# # Write results of model selection
# fwrite(table0, file = "South Carolina/2021/best_model_selection0.csv", quote = FALSE,
#        row.names = FALSE)



#define formulas by updating the baseline formula with different combinations of cross-basis functions

# f1.1 <- update.formula(baseformula4, ~. + basis_heat)
# f1.10 <- update.formula(baseformula4, ~. + Vheat)
# f1.2 <- update.formula(baseformula4, ~. + basis_prec)
# f1.20 <- update.formula(baseformula4, ~. + Vprec)
# f1.3 <- update.formula(baseformula4, ~. + basis_air)
# f1.30 <- update.formula(baseformula4, ~. + Vair)
# f1.4 <- update.formula(baseformula4, ~. + basis_Stringency)
# f1.40 <- update.formula(baseformula4, ~. + Vstring)
# 
# # create a list of formulas//for period without covid use the second one
# formulas <- list(f1.1,f1.10,f1.2,f1.20,f1.3,f1.30,f1.4,f1.40)
# formulas <- list(f1.1,f1.10,f1.2,f1.20,f1.3,f1.30) 
# formulas <- list(f1.4,f1.40)
# # create model label string
# lab <- c("model_1.1","model_1.10","model_1.2","model_1.20","model_1.3","model_1.30","model_1.4","model_1.40")
# lab <- c("model_1.1","model_1.10","model_1.2","model_1.20","model_1.3","model_1.30")
# lab <- c("model_1.4","model_1.40")
# 
# # create a function to run a model for each formula in the list and save the model output to file
# # WARNING: this may take a long time to run
# models <- lapply(1:length(formulas),
#               function(i) {
#                 model <- mymodel(formula = formulas[[i]], data = df, family = "Gaussian", config = FALSE)
#                 save(model, file = paste0("South Carolina/2021/", lab[i],".RData"))})
# 
# # create table to store DIC and select best model
# table1 <- data.table(Model  =  c( "heatlag","heat","preclag","prec","airlag","air","stringencylag","stringency"),
#                      DIC = NA,
#                      logscore = NA)
# table1 <- data.table(Model  =  c( "heatlag","heat","preclag","prec","airlag","air"),
#                      DIC = NA,
#                      logscore = NA)
# 
# for(i in 1:length(formulas))
#   {
#   load(paste0("South Carolina/2021/",lab[i],".RData"))
#   table1$DIC[i] <- round(model$dic$dic, 2)
#   table1$logscore[i] <- round(-mean(log(model$cpo$cpo), na.rm = T), 3)
# }
# 
# # view table
# table1
# 
# # define position of best fitting model
# best.fit <- which.min(table1$DIC)
# 
# # Write results of model selection
# fwrite(table1, file = "South Carolina/2021/best_model_selection1.csv", quote = FALSE,
#        row.names = FALSE)


#define formulas by updating the baseline formula with different combinations of cross-basis functions

f2.1 <- update.formula(baseformula5, ~. + basis_air+basis_prec)
f2.2 <- update.formula(baseformula5, ~. + basis_air+basis_heat)
f2.3 <- update.formula(baseformula5, ~. + basis_air+basis_prec+basis_heat)


# create a list of formulas//for period without covid use the second one
formulas <- list(baseformula5,f2.1,f2.2,f2.3)

# create model label string
lab <- c("base5","model_2.1","model_2.2","model_2.3")

# create a function to run a model for each formula in the list and save the model output to file
# WARNING: this may take a long time to run
models <- lapply(1:length(formulas),
                 function(i) {
                   model <- mymodel(formula = formulas[[i]], data = df, family = "Gaussian", config = FALSE)
                   save(model, file = paste0("alabama/", lab[i],".RData"))})

# create table to store DIC and select best model
table2 <- data.table(Model  =  c( "base","air+prec","air+heat","aire+prec+heat"),
                     DIC = NA,
                     logscore = NA)

for(i in 1:length(formulas))
{
  load(paste0("alabama/",lab[i],".RData"))
  table2$DIC[i] <- round(model$dic$dic, 2)
  table2$logscore[i] <- round(-mean(log(model$cpo$cpo), na.rm = T), 3)
}

# view table
table2

# define position of best fitting model
best.fit <- which.min(table2$DIC)

# Write results of model selection
fwrite(table2, file = "alabama/best_model_selection2.csv", quote = FALSE,
       row.names = FALSE)


