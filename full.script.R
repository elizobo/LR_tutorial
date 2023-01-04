# Logistic regression tutorial
# Elizabeth Stroud; s1828407@ed.ac.uk
# Date completed : 03/01/2023

# SETUP ----

# Set the working directory
setwd("your_filepath")

# Load your libraries

library(ggplot2)
library(dplyr)
library(corrplot)
library(MuMIn)
library(predictmeans)
library(InformationValue)
library(caret)
library(sjPlot)


# Set a plot theme

# Define a custom plot theme
plot_theme <- function(...){
  theme_classic() +
    theme(                                
      axis.text = element_text(size = 7,                           # adjust axes
                               colour = "black"),
      axis.text.x = element_text(margin = margin(5, b = 10)),
      axis.title = element_text(size = 6,
                                colour = 'black'),
      axis.ticks = element_blank(),
      plot.background = element_rect(fill = "white",               # adjust background colors
                                     colour = NA),
      panel.background = element_rect(fill = "white",
                                      colour = NA),
      legend.background = element_rect(fill = NA,
                                       colour = NA),
      legend.title = element_text(size = 10),                      # adjust titles
      legend.text = element_text(size = 16, hjust = 0,
                                 colour = "black"),
      plot.title = element_text(size = 17,
                                colour = 'black',
                                margin = margin(10, 10, 10, 10),
                                hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5,
                                   colour = "black",
                                   margin = margin(0, 0, 30, 0)),
      plot.caption = element_text(plot.caption = element_text(size = 50,
                                                              hjust=0))
      )
}



# Load data
cones <- read_csv("data/cones.csv")
summary(cones) # Look at general structure of the data


# CHECK ASSUMPTIONS ----

## Assumption 1
# Look at data structure

head(cones) # print first 6 rows of data
tail(cones) # print last 6 rows of data


## Assumption 2
# Make abundance into P/A data

conesbi <- cones %>% 
  dplyr::mutate( 
    Presence = (case_when(  # making new column Presence 
                 Count > '0' ~ 1,       # making each measurement where the Count is over 0 a cone presence
                 Count == '0' ~ 0 )))   # making each measurement where the Count is 0 a cone absence


## Assumption 3
# Test continuous predictor variable autocorrelation

cov(conesbi$Age, conesbi$DBH) # print covariation value (indicates direction not strength)
correlations <-  cor(conesbi[,4:5]) # extract continuous predictors for correlation test
correlations # print correlation value (closer to 1 or -1 indicates strong +ve or -ve value)
corrplot(correlations, method="circle") # create circle plot showing correlation

## Assumption 4
# Check outliers of the explanatory variable

boxplot(conesbi$DBH, main = "Boxplot") # any circles outside the whiskers indicate outliers

## Assumption 5
# Test linear explanatory~logit(response) relationship

boxTidwell(Presence ~ DBH, data = conesbi) # test treesize cone presence relationship
# highly significant correlation

## Assumption 6
# Checking data quantity for explanatory variables

obs <- length(conesbi$Presence) # find number of observations
summary(conesbi$Presence) # find least frequent outcome and its proportion.
abs_lik <- (1 - (mean(conesbi$Presence))) # find likelihood of cone absence
max_e_vars <- obs * abs_lik / 10
max_e_vars # print the maximum number of explanatory variables allowed in the model



# EXPLORE DATA ----

# Boxplot to compare distributions of binomial distribution with continuous predictor

data_dist <- melt(conesbi[, c("Presence", "DBH")], # you can add in all your possible continuous predictor variables to have a look at the data distribution with it
                  id.vars="Presence") # set your response variable

ggplot(data_dist, aes(factor(Presence), y = value, fill=factor(Presence))) +
  geom_boxplot() + 
  facet_wrap(~variable, scales="free_y") + # make panels of plots for each variable with a y scale that can be used for different units
  labs(caption = '\n Figure 1: Looking at data distribution of cone presence with Diameter at Breast Height (DBH)\n for conifers in the southern Rocky Mountains, USA ') + 
  plots_theme() +
  theme(plot.caption = element_text(size = 30,
                                    hjust = 0),
        legend.position = "right",
        legend.direction = "vertical",
        legend.title =  element_text(size = 30),
        legend.text = element_text(size = 30))

ggsave(filename = 'images/distrib1.png',
       width = 20,
       height = 15, 
       units = 'in')

# Binomial distribution with categorical on continuous predictor

ggplot(conesbi, aes(x = DBH, y = Presence, colour = Spec)) + 
  geom_point(size = 10, alpha = .2) + # make data points transparent so they can be seen overlayed
  labs(caption = '\n Figure 2: Looking at data distribution of cone presence with Diameter at Breast Height (DBH)\n for Engelmann spruce (PIEN, Picea engelmannii) and subalpine fir (ABLA, Abies lasiocarpa) in\n the southern Rocky Mountains, USA ') + 
  plots_theme() +
  theme(plot.caption = element_text(size = 30,
                                    hjust = 0),
        legend.position = "right",
        legend.direction = "vertical",
        legend.title =  element_text(size = 30),
        legend.text = element_text(size = 20))

ggsave(filename = 'images/distrib2.png',
       width = 20,
       height = 15, 
       units = 'in')



# MAKE MODELS ----
# How does the likelihood of conifer cone presence change with tree age and species?

# Null - make a null model to compare the others to

null.mod <- glm(Presence ~ 1,  # there are no explanatory variables
                data = conesbi,
                family = binomial(link = "logit")) # setting the family as a binomial distribution but with a logit link to make it linear

#summary(null.mod) # print a model summary if you want to see whats at play


# Make a model with fixed and random effects 
dbh.mod <- glmer(Presence ~ DBH + Spec + (1 | Plot / ID) + (1 | Year), 
                 data = conesbi,
                 family = binomial(link = "logit"))
#summary(dbh.mod) 


# Make a model with interacting fixed effects 
dbh.mod.int <- glmer(Presence ~ DBH * Spec + (1 | Plot / ID) + (1 | Year), 
                     data = conesbi,
                     family = binomial(link = "logit"))

#summary(dbh.mod.int) # you can print a summary of your models if you'd like and compare their outcomes


# Compare ur models with AICc galfriend!
AICc(null.mod, dbh.mod, dbh.mod.int)
## species can't be included as a random effect because only 2 levels
# but seems like its better included as a fixed effect than not at all


# TEST YOUR MODEL ----

# Check outliers with Cook's distance plot
model <- glm(Presence ~ DBH + Spec, family = "binomial", data = conesbi)
CookD(model) # this might pop up in another window


# Test and train then model

# Make random sample reproducible (so you can come back to this and get the same set of numbers in the training and testing dataset)
set.seed(1) # you can use any integer here

# Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(conesbi), replace=TRUE, prob=c(0.7,0.3)) # randomly sample 70%
train <- conesbi[sample, ]  # train dataset including 70% of the data
test <- conesbi[!sample, ]  # make the test dataset with the remaining 30%

# Make a training model using the training data
train.mod <- glmer(Presence ~ DBH + Spec + (1 | Plot) + (1 | Year), # ID can't be included otherwise numbers wont be present for both training and testing datasets
                 data = train,
                 family = binomial)


# Use your training model to make predictions for the test data
pdata <- predict(train_mod, newdata = test, type = "response")

# Compute a confusion matrix comparing the predicted model outcomes to the real outcomes for the test dataset
confusionMatrix(data = as.factor(as.numeric(pdata>0.5)), reference = as.factor(test$Presence))  # the extracted predictions are probabilities and must be set as numerical 1/0 factors to match the original test data
# Calculate total misclassification error rate
optimal <- optimalCutoff(test$Presence, p_data())[1]
misClassError(test$Presence, predicted, threshold=optimal)


# Make predictions
new <- data.frame(DBH = 10, Spec  = "ABLA", Plot = "1", Year = "2018")  # create a new data frame defining predictor variables
predict(train.mod, new, type="response")  # print model predicted probability of it producing cones


# PLOT MODEL PREDTICTIONS ----

# Look at the model outcomes
summary(dbh.mod)
# Print a table for reporting values
tab_model(dbh.mod)
# Make fixed effects plots showing the odds ratio
plot_model(dbh.mod,
           type = "est", 
           show.values = TRUE)


# Calculate CI
dbhCI <- 1.16*

# Now we know the models are pretty good we can plot their prediction

# Extract predictions
predicted_dbh <- plot_model(dbh.mod, type = "pred", terms = "DBH [all]")$data # extract model predictions in response to DBH

# Make your marginal effects plots
predicted_dbh_plot <- ggplot(data = predicted_dbh) + 
  geom_line(aes(x = x, # add a line showing the model's prediction
                y = predicted),
            colour = '#009E73',
            linewidth = 1) + 
  geom_ribbon(aes(x = x, # add a line to show 95% confidence intervals
                  ymin = conf.low,
                  ymax = conf.high), 
              fill = "#009E73",
              alpha = 0.3) + 
  labs(x = 'Tree DBH (cm)', # personalise labels
       y = 'Probability of cone presence',
       caption = '\n Figure 1: Probability of cone presence predicted from tree size for conifers in the \n southern Rocky Mountains, USA (+- 95% CI). ') + 
  plots_theme() +
  theme(plot.caption = element_text(size = 50, # for come reason this caption stuff just won't come through from the function we made so you have to reiterate it here
                                    hjust = 0))

# Save your plot
ggsave(filename = 'images/p_cone_dbh.png',
       width = 30,
       height = 15, 
       units = 'in')




# Make a plot with separate predictions for each categorical variable

# Extract predictions again
predicted_dbh_sp <- plot_model(dbh.mod, type = "pred", terms = c("DBH [all]", "Spec"))$data # extract model predictions in response to DBH and species

# Make your marginal effects plots
predicted_dbh_sp_plot <- ggplot(data = predicted_dbh_sp) + 
  geom_line(aes(x = x, # add a line showing the model's prediction
                y = predicted, 
                col = group_col), # colour the lines for each species
            linewidth = 1) + 
  geom_ribbon(aes(x = x, # add a line to show 95% confidence intervals
                  ymin = conf.low,
                  ymax = conf.high,
                  fill = group_col), 
              alpha = 0.3) + 
  labs(x = 'Tree DBH (cm)', # personalise labels
       y = 'Probability of cone presence',
       caption = '\n Figure 2: Probability of cone presence predicted from tree size for two conifer species\n in the southern Rocky Mountains, USA (+- 95% CI). ',
       ) + 
  plots_theme() +
  scale_colour_manual(name = "Conifer species", labels = c("Subalpine fir", "Engleman spruce"), values = c("#E69F00", "#CC79A7")) +  # personalise colours and legend names
  scale_fill_manual(name = "Conifer species", labels = c("Subalpine fir", "Engleman spruce"), values = c("#E69F00", "#CC79A7")) +
  theme(plot.caption = element_text(size = 50, # for come reason this caption and legend stuff just won't come through from the function we made so you have to reiterate it here
                                    hjust = 0),
        legend.position = "right",
        legend.direction = "vertical",
        legend.title =  element_text(size = 40),  # alter size and location of legend to suit your plot
        legend.text = element_text(size = 35))

            

# Save your plot
ggsave(filename = 'images/p_cone_dbh_sp.png',
       width = 30,
       height = 15, 
       units = 'in')


