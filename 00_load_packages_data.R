## R script to prepare data and lagged variables for INLA-DLNM modelling
rm(list=ls())
setwd('~/Desktop/sisi/')
#  select other packages
library(INLA)
packages <- c("data.table", "tidyverse", "sf", "sp", "spdep",
              "dlnm", "tsModel", "hydroGOF","RColorBrewer", 
              "geofacet", "ggpubr", "ggthemes")

# install.packages
# lapply(packages, install.packages, character.only = TRUE)

# load packages
lapply(packages, library, character.only = TRUE)



data<-read_csv("~/Desktop/sisi/ModellingData.csv")

data<-data[data$STATEcode==48,]

## load data
# Climate and case/intervention/holiday data in the first week for lagged values 

# load shape file 
map_all <- read_sf("~/Desktop/sisi/tl_2023_us_county/tl_2023_us_county.shp")
data$GEOID <- as.character(data$GEOID)
#data$GEOID <- paste0("0", data$GEOID)
map <- map_all[map_all$GEOID %in% unique(data$GEOID),]
#dim(map)

# Create adjacency matrix
sf::sf_use_s2(FALSE) # to fix 4 features with invalid spherical geometry
nb.map <- poly2nb(as_Spatial(map$geometry))
g.file <- "texas/map.graph"
if (!file.exists(g.file)) nb2INLA(g.file, nb.map)


# load pre-defined grid of Chinese provinces for geofacet plots
# grid <- read.csv("data/Provinces.csv")

## integrate data
# Create lagged variables
# define matrices of lagged terms for climate, covid and intervention variables

# set max lag - by day
nlag<-14


lag_heat<-tsModel::Lag(data$HeatCount, group = data$GEOID, k = 0:nlag)
lag_prec <- tsModel::Lag(data$Precipitation, group = data$GEOID, k = 0:nlag)
lag_airpoll <- tsModel::Lag(data$AirPolllution, group = data$GEOID, k = 0:nlag)
lag_stringency <- tsModel::Lag(data$StringencyIndex, group = data$GEOID, k = 0:nlag)

lag_heat <- lag_heat[data$Week>20,]
lag_prec <- lag_prec[data$Week>20,]
lag_airpoll <- lag_airpoll[data$Week>20,]
lag_stringency <- lag_stringency[data$Week>20,]
# remove year 2000 from dengue dataframe
data <- data[data$Week> 20,]
## define dimensions
# re-define time indicator 
unique(data$Date)
# data$time <- data$time - 7

# total number of days
nday <- length(unique(data$Date))
# total number of weeks
data$Date <- as.Date(data$Date, format = "%m/%d/%Y")
data <- data %>%
  mutate(week2 = as.integer(difftime(Date, min(Date), units = "weeks")) + 1)

nweek <- length(unique(data$week2))

# total number of cities
ncity <- length(unique(data$GEOID))
# total number of provinces
#nprov <- length(unique(data$prov_code))

# define cross-basis matrix (combining nonlinear exposure and lag functions)
# set lag knots

lagknot = equalknots(0:nlag, 2)
# Temporature / fun 'ns' - Generate a Basis Matrix for Natural Cubic Splines
var <- lag_heat
basis_heat <- crossbasis(var,
                    argvar = list(fun = "ns", knots = equalknots(data$HeatCount, 2)),
                    arglag = list(fun = "ns", knots = nlag/2))
head(basis_heat)


var <- lag_prec
basis_prec <- crossbasis(var,
                         argvar = list(fun = "ns", knots = equalknots(data$Precipitation, 2)),
                         arglag = list(fun = "ns", knots = nlag/2))
head(basis_prec)


var <- lag_airpoll
basis_air <- crossbasis(var,
                         argvar = list(fun = "ns", knots = equalknots(data$AirPolllution, 2)),
                         arglag = list(fun = "ns", knots = nlag/2))
head(basis_air )


var <- lag_stringency
basis_Stringency <- crossbasis(var,
                                argvar = list(fun="ns", knots = equalknots(data$StringencyIndex, 2)),
                                arglag = list(fun="ns", knots = lagknot))
head(basis_Stringency)

##test linear interaction with urbanization rate
# centre the variable at different levels of density (25th, 50th and 75th percentiles)
# from high to low rate



# ## test linear interaction with GDP per capita
# # centre the variable at different levels of GDP (25th, 50th and 75th percentiles)
# # from high to low 
 
# summary(data$gdp_pc_2019)
# # set indicator to zero at point of interest (centring point)
# # re-parameterise model to extract different predictions

# ## Multiply each cross-basis variable by the linear terms (see Gasparrini et al. EHP 2015)
# # note: exploit the column by column product 



# # multiply the covid intervention cross-basis variables by the GDP per capita linear terms



# assign unique column names to cross-basis matrix for inla() model
# note: not necessary for glm(), gam() or glm.nb() models
colnames(basis_heat) = paste0("basis_heat.", colnames(basis_heat))
colnames(basis_prec) = paste0("basis_prec.", colnames(basis_prec))

colnames(basis_air) = paste0("basis_air.", colnames(basis_air))
colnames(basis_Stringency) = paste0("basis_Stringency.", colnames(basis_Stringency))

# note: for INLA models an index should start with 1 and with the max value equal to the length of unique values

# create city index 

data$county_index <- rep(1:ncity, nday)


# create province index
# state length
k <- unique(data$STATEcode)

for (j in 1:1){
  data$STATEcode[data$STATEcode == k[j]] <- j 
}

# create week index
# set first week for modelling to 1
data$week_index <- data$week2

data$weekday<- as.integer(format(data$Date, "%u"))

#### set up data and priors for INLA model
## set data for models - try intraflow first, then in_outflow
Y  <- data$SentimentScore # response variable
N  <- length(Y) # total number of data points
# E  <- data$pop_2019 # model offset 
# random variable
T1 <- data$weekday # for random effect to account for day-of-week effect
T2 <- data$week_index # for random effect to account for inter-week variability
S1 <- data$county_index # for city-level spatial random effect
S2 <- data$STATEcode # for provincial interaction with daily random effect
# Other variables
# Vu <- data$urban # include urbanization rate variable along with linear urbanization rate interaction
Vv <- data$VulnerabilityIndex# include GDP per capita along with linear GDP interaction
Vheat<-data$HeatCount
Vprec<-data$Precipitation
Vair<-data$AirPolllution
Vstring<-data$StringencyIndex
Vh <- data$Holiday

# create dataframe for model testing
df <- data.frame(Y, T1, T2, S1, S2, Vv, Vh,Vheat,Vprec,Vair,Vstring)

# define priors

precision.prior2 <- list(prec = list(prior = "pc.prec", param = c(1, 0.01)))
# inla model function

# include formula and set defaults for data, family (to allow other prob dist models e.g. Poisson) and config (to allow for sampling)
mymodel <- function(formula, data = df, family = "Gaussian", config = FALSE)
  
{
  model <- inla(formula = formula, data = data, family = family, 
                control.inla = list(strategy = 'adaptive',int.strategy='eb'), 
                control.compute = list(dic = TRUE, config = config, 
                                       cpo = TRUE, return.marginals = FALSE),
                control.fixed = list(correlation.matrix = TRUE, 
                                     prec.intercept = 1, prec =1),
                control.predictor = list(link = 1, compute = TRUE), 
                verbose = FALSE)
  model <- inla.rerun(model)
  return(model)
}

