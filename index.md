
# Binomial Logistic Regression Tutorial

<p align="center">
   <img src="{{ site.baseurl }}/images/subalp_forest.png" alt="drawing" width="75%">
</p>
<p align="center">
  Subalpine fir forest (photo credit: <a href="https://www.flickr.com/photos/codiferous/7978232221/in/photostream/" target="_blank">C. Hinchliff</a>).
</p>


### Tutorial Aims

#### <a href="#section1"> 1. Check assumptions of binomial logistic regression</a>

#### <a href="#section2"> 2. Build a binomial logistic regression model</a>

#### <a href="#section3"> 3. Test a binomial logistic regression model</a>

#### <a href="#section4"> 4. Present and report the results of a binomial logistic regression model</a>

## Basics for working with the binomial

We often have work with binary data in ecology. Whether measuring germination success, the presence of a species in a quadrat, or an animal's behavior choice, we are left with a binary response variable. To see how this binary is affected by a continuous variable (often an environmental gradient) we can carry out logistic regression.

__All the B-word Terminology__

-   __Binary__ : Your data is binary if it has 2 outcomes. For example, left or right, pink or white, success or failure, presence or absence, yes or no. You can always represent these outcomes as '0' and '1'.
<p align="center">
   <img src="{{ site.baseurl }}/images/binomial.png" alt="drawing" width="30%">
</p>

-   __Boolean__ : Your data is Boolean if you have combination outcomes you can define as binary data with values of true and false. You pretty much only have to think of data in this way if you're doing Boolean Algebra - building a deductive logical system (not part of this tutorial, phew).
<p align="center">
   <img src="{{ site.baseurl }}/images/bernoulli1.png" alt="drawing" width="35%"/>&nbsp;&nbsp;&nbsp;&nbsp;
   &nbsp;&nbsp;&nbsp;&nbsp;<img src="{{ site.baseurl }}/images/bernouli2.png" alt="drawing" width="50%">
</p>

-   __Bernoulli__ : A Bernoulli trial/experiment is a single binary experiment. The outcome of this has Bernoulli distribution - the observed response of '0' or '1' from a single trial.
<p align="center">
   <img src="{{ site.baseurl }}/images/beta.png" alt="drawing" width="30%">
</p>

-   __Binomial__ : A binomial distribution describes the outcome of several Bernoulli trials - the probability that in X number of trials there will be Y number of '1' outcomes.
<p align="center">
   <img src="{{ site.baseurl }}/images/actualbeta.png" alt="drawing" width="60%">
</p>

-   __Beta__ : A beta distribution also describes the outcome of several Bernoulli trials but as a probability of having '1' as an outcome, given the number of '1' and '0' outcomes from X number of trials. As the number of Bernoulli trials increases the beta distribution will form an increasingly arched bell shape.
<p align="center">
   <img src="{{ site.baseurl }}/images/PDF-graph.png" alt="drawing" width="50%">
</p>



Binomial logistic regression is a type of Generalised Linear Model (GLM). You can always give yourself a refresher on model building with <a href="(https://ourcodingclub.github.io/tutorials/model-design/)" target="_blank">this Coding Club tutorial</a>. If you have time and are interested below are some questions which although you aren't asking yourself, may make the whole binomial logistic regression concept a bit clearer for the tutorial ahead. If you aren't curious about the ins and outs, trot blindly past these questions on to the practical stuff. 

{% details Click to expand %}

	~~~ python
	Code here
	~~~
{% enddetails %}

- <details>
	<summary>Why can't we use linear regression?</summary>

	<pre>
	     - Well the assumptions of linear regression that a) residuals are normally distributed and b) the response variable is a continuous and unbounded ratio or interval value are both violated with this categorical binary response variable.
	     - If we used our binary outcomes as the response variable (Y-axis on a graph) and fit a straight line, this doesn't represent the relationship very well. 
	     - Coding Club has <a href="(https://ourcodingclub.github.io/tutorials/mixed-models/)" target="_blank">tutorials on linear regression</a> if you want to know more about them. 
	</pre>

</details>

- <details>
	<summary>Um logit link function??</summary>

	<pre>
	     - A link function is function of the mean of the response variable (Y-axis) that we use as the response (Y-axis) instead of response variable itself. So we use the logit of the response variable (Y-axis) instead of just the response variable. 
	     - The logit function is the natural log of the odds that the response will equal 1. 
	</pre>

</details>

- <details>
	<summary>Why binomial?</summary>

	<pre>
	     - Logistic regression refers to any regression model in which the response variable is categorical.   
	     - Binomial logistic regression deals with binary categorical response variables, but other types of logistic regression can deal with more than 2 categories.   
	     - Multinomial logistic regression: Deals with response variables with three or more categories with no natural ordering among the categories (e.g. a hat trick producing nothing, a rabbit or stars).   
	     - Ordinal logistic regression: The response variable can belong to one of three or more categories and there is a natural ordering among the categories (e.g. a hat trick producing a rabbit with white, white and black spotted or black fur).
	</pre>

</details>
	
- <details>
	<summary>What is maximum likelihood estimation?</summary>
	
	<pre>
	     - Logistic regression uses maximum likelihood estimation to fit a model.  
	     - In maximum likelihood estimation a set of parameters is chosen for a model that maximizes a likelihood function.
	</pre>

</details>



<br/>

<p align="center">
   <img src="{{ site.baseurl }}/images/logit.png" alt="drawing" width="80%">
</p>
<p align="center">
  Working with binary data: a) the inability to represent a the relationship of binomial data linearly (left), b) the ability to linearly represent it if using log-odds of presence as the response (centre) and c) using a logit link function to convert the response to probability, forming a sigmoidal relationship. 
</p>

If you want to go into more of the maths have a read <a href="[https://www.flickr.com/photos/codiferous/7978232221/in/photostream/](https://medium.com/deep-math-machine-learning-ai/chapter-2-0-logistic-regression-with-math-e9cbb3ec6077)" target="_blank">here</a>, otherwise this tutorial can give you all you need to get your report underway from raw data to results.

<br/>

### Data: Conifer cones

You can get all of the resources for this tutorial from <a href="https://github.com/elizobo/LR_tutorial" target="_blank">this GitHub repository</a>. Clone and download the repo as a zip file, then unzip it.

For this tutorial we'll be looking into reproductive maturity of conifers. There is high variability in seed cone production among many northern temperate conifers, and we'll make a model to find reasons for this variation.

<p align="center">
  <img src="{{ site.baseurl }}/images/engelman_cone2.png" alt="drawing" width="30%"/>&nbsp;&nbsp;&nbsp;&nbsp;
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="{{ site.baseurl }}/images/subalpfir_cone.png" alt="drawing" width="35%" > 
</p>
<p align="center">
  Engleman spruce seed cones (photo credit: <a href="https://https://www.conifers.org/pi/Picea_engelmannii.php" target="_blank"> C. Earle </a>).&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Subalpine fir seed cones (photo credit: <a href="https://www.flickr.com/photos/76416226@N03/6881892262" target="_blank"> B. Leystra </a>). 
</p>


Our dataset is a mixture of cone abundance from Subalpine fir, _Abies lasiocarpa_, and Engleman spruce, _Picea engelmannii_ from southern Rocky Mountains, USA. The data is from this <a href="https://portal.edirepository.org/nis/home.jsp" target="_blank"> neat open source database site </a>.


First we'll set up the RStudio working environment and load in this data.
```r
## Set up----

# Set the working directory

setwd("your_filepath")


# Load your libraries

library(dplyr)
library(corrplot)
library(MuMIn)
library(predictmeans)
library(InformationValue)
library(caret)
library(InformationValue)
library(car)
library(reshape2)
library(sjPlot)


# Set a plot theme function

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
summary(cones)  # look at general structure of the data
```

The summary() function displays a summary statistics of data for each column:

-  __Plot__: Unique number assigned to each tree plot

-  __Tree__: Number assigned to each tree in a plot

-  __ID__: Unique number identifier for each tree

-  __Spec__: Tree species: ABLA (Subalpine fir) or PIEN (Engleman spruce)

-  __DBH__: Tree Diameter at Breast Height measurement (cm)

-  __Age__: Tree age measured at the base of the tree in 2016

-  __Year__: The year of data collection

-  __Count__: Estimate of tree seed cone abundance

What effects conifer reproductive maturity? If cone presence is an indicator of reproductive maturity, what are the predictor and response variables available here to answer our question?

<a name="section1"></a>

<pre>


</pre>

## 1.  Check assumptions of binomial logistic regression

Before making a logistic regression model you have to check your data is suited for it. There are 6 assumptions we'll work through.

__Assumption 1. Observations are independent__

Each data point used to construct the model must be equally related. To check this we will print the first and last 6 rows of the data and check any metadata we have available.

```r
# Look at data structure

head(cones)  # print first 6 rows of data
tail(cones)  # print last 6 rows of data
```

From this we can gather that 1 measurement is taken every year for 3 years for each tree and there are multiple trees per plot. This shows that not all data points are independent. Some measurements will be more linked than others; some are from the same tree, some trees are in the same plot, and some are taken on the same year.

To account for these links and avoid pseudo replication we will have to account for them as random effects. You incorporate variables as random effects when they are not the main focus of a study but may impact the dependent variable. Random effects we'll need to include:

-   _treeID nested within Plot_ : We aren't interested in how individual trees or trees within different plots differ, but these variables will impact the cone presence. This is known as a hierarchical random effect because it takes into account the structured clustering of the data; a tree within a plot.

-   _year_ : The year data was recorded may effect the cone presence measurements. For example, if one of the years was particularly hot there may be delayed cone production, reducing the number of trees recorded with cone presence. We aren't interested in this effect of year but we will take it's variation into account by making it a random effect.
 
<p align="center">
   ASSUMPTION MET: If we take these variables into account as random effects, we can assume datapoints are independent.
</p>
<br/>
<br/>
__Assumption 2. The response variable is binomial__

Currently we only have *cone abundance* data. We will make a new response variable column, *cone presence*. Presence will be indicated by 1 and absence by 0.

```r
# Make abundance into P/A data

conesbi <- cones %>% 
  dplyr::mutate( 
    Presence = (case_when(  # making new column Presence 
                 Count > '0' ~ 1,       # making each measurement where the Count is over 0 a cone presence
                 Count == '0' ~ 0 )))   # making each measurement where the Count is 0 a cone absence
```
<p align="center">
   ASSUMPTION MET: We have a binomial response variable.
</p>
<br/>
<br/>
__Assumption 3. Predictor variables are independent with no multicolinearity__

Predictor variables included in the model as fixed effects must be independent. Non of the predictor variables can be related to any of the others otherwise they will explain the same variation in the response variable, and the model will appear more powerful than it actually is.

We will test to see if there is any correlation between them by making a correlation plot. The larger and darker the circle, the stronger the correlation between the variables. Blue colouring indicated a positive correlation and red indicated a negative correlation.

```r
# Test continuous predictor variable autocorrelation

cov(conesbi$Age, conesbi$DBH) # print covariation value (indicates direction not strength)
correlations <-  cor(conesbi[,4:5]) # extract continuous predictors for correlation test
correlations # print correlation value (closer to 1 or -1 indicates strong +ve or -ve value)
corrplot(correlations, method="circle") # create circle plot showing correlation
```
<p align="center">
   <img src="{{ site.baseurl }}/images/corrplot.png" alt="drawing" width="50%">
</p>
<p align="center">
  Correlation plot for the models potential explanatory variables.
</p>

There is a very strong positive relationship between tree age and DBH. Including them both in this model would violate this assumption, so they must be incorporated in models separately.

<p align="center">
   ASSUMPTION MET: We will use DBH as the only fixed effect in this model so that it is an independent explanatory variable.
</p>
<br/>
<br/>
__Assumption 4. There are no extreme outliers in the explanatory variable.__

Data points that are extreme outliers have a disproportionate effect on the data trend and may affect the prediction probability of the datapoints by the model. Although the sigmoid function in logistic regression models tapers outliers so they aren't as influential as they would be in other model types, extreme outliers may still lower the model performance so should be identified and removed ONLY IF there is any valid reason for exclusion (e.g. it turned out the tree was actually dead).

We will get an idea of any potential outliers by seeing if we have a good representation of data along the range of our explanatory variable. We will get an idea of any outliers by creating a boxplot of our response variable. R will highlight any datapoints outside 1.5 times the interquartile range as circles.

```r
# Check outliers of the explanatory variable

boxplot(conesbi$DBH, main = "Boxplot")
```

<p align="center">
   <img src="{{ site.baseurl }}/images/outlier_box.png" alt="drawing" width="50%">
</p>
<p align="center">
  Distribution of explanatory Diamater at Breast Height data with outliers highlighted as circles.
</p>

We can see there is a general under representation of older trees which may lead to outliers of our model. We will keep this in mind and check this assumption again using cooks distance test once we've built our model.
<br/>
<br/>
__Assumption 5. There is a linear relationship between the explanatory variable and the logit of the response variable__

To ensure the data relationship fits a binomial distribution there must be a linear relationship between the explanatory variable and the logit of the response variable. The logit function describes the non-linearity - the S-shape (sigmoid curve) - seen in logistic regression curves, so by applying to logit function it effectively linearises the relationship for our generalised linear model. 

We will test that the explanatory and response have this sigmoid this relationship through a Box Tidwell test.

```r
# Test linear explanatory~logit(response) relationship

boxTidwell(Presence ~ DBH, data = conesbi)
```

If the p-value (Pr(>[z]) is significant (below 0.05, also indicated by asterisks) the data fits the relationship; this test shows the correlation between presence and age will fit the sigmoid function of logistic regression.

<p align="center">
  ASSUMPTION MET: There is a linear relationship between the explanatory and logit of the response.
</p>
<br/>
<br/>
__Assumption 6. There is a sufficiently large sample size__

The model must contain a minimum of 10 observations of the least frequent outcome for each explanatory variable.

We will find the least frequent outcome and work out how many explanatory variables we are allowed for this model.

```r
# Checking data quantity for explanatory variables

obs <- length(conesbi$Presence) # find number of observstions
summary(conesbi$Presence) # find least frequent outcome and its proportion.
abs_lik <- (1 - (mean(conesbi$Presence))) # find likelihood of cone absence
max_e_vars <- obs * abs_lik / 10
max_e_vars # print the maximum number of explanatory variables allowed in the model
```

-   There are 1426 observations.

-   The Presence mean is over 0.5, meaning the majority of observation outcomes are cone presence (represented by 1 in the binary). This means absence is the least frequent outcome.

-   A maximum of 62 explanatory variables are allowed in the model.

<p align="center">
  ASSUMPTION MET: We will include less than 62 explanatory variables in the model.
</p>
<br/>
<br/>
__Great! With all these assumptions met we know our data is suitable for a logistic regression and we can get along with building our model.__


<a name="section2"></a>

## 2.  Build a binomial logistic regression model

Before building your model, remind yourself of your research question :

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;__How does tree age and species influence conifer reproductive maturity?__

Make sure you know your response variable, explanatory variable(s) and random effects and their names within the model.
   
| __Model parameter type__  | __Our model parameter__                                          | __Name in Rscript__ |
|---------------------------|------------------------------------------------------------------|---------------------|
| __Response variable__     | - conifer reproductive maturity, indicated by seed cone presence | - Presence          |
| __Explanatory variables__ | - tree size                                                      | - Age               |
|                           | - conifer tree species                                           | - Spec              |
| __Random effects__        | - individual tree                                                | - ID                |
|                           | - plot of trees                                                  | - Plot              |
|                           | - measurement year                                               | - Year              |




#### __Check out the data__

If you're not in a rush you can have a look at your data distribution to get an idea of what you're working with.

Create a boxplot to see the trend in binomial data against your continuous predictor(s). You can add more predictor variables into the code below if you want to compare their relative distribution.

```r
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
```


<p align="center">
   <img src="{{ site.baseurl }}/images/distrib1.png" alt="drawing" width="70%">
</p>
<p align="center">
  Distribution plot for the continuous explanatory variable.
</p>

This plot shows there are more cases of cone presence for larger trees but there is quite a lot of overlap with cases of cone presence and absence for the full range of DBH values.

We'll see how the binary is distributed between a categorical explanatory variable as well.

```r
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
```

<p align="center">
   <img src="{{ site.baseurl }}/images/distrib2.png" alt="drawing" width="70%">
</p>
<p align="center">
  Distribution plot for continuous and categorical explanatory variables.
</p>

The Engleman spruce appears to have more observations of cone presence than Subalpine fir, but there is a lot of overlap and no obvious trend for either species. There are also fewer observations for larger than smaller trees and the size range of Subalpine fir does not range as large as Engleman spruce. This is likely to make model predictions of larger trees less trustworthy as it is based on less data.

<pre>


</pre>

#### __Modelling__

First we'll make a null model. This plots the response against a constant to see the strength of model predictions from chance alone.

```r
# Null - make a null model to compare the others to

null.mod <- glm(Presence ~ 1,  # there are no explanatory variables
                data = conesbi,
                family = binomial(link = "logit")) # setting the family as a binomial distribution but with a logit link to make it linear


#summary(null.mod)  # you can print a summary of your models if you'd like and compare their outcomes
```

Now we'll make up our more complex models including the explanatory variables and random effects. See how we've nested the tree ID within Plot to account for the structure of the data.

We'll try including our explanatory variables as separate fixed effects and interacting fixed effects. Whilst the separate (+) fixed effects allow the effect of tree species and DBH on cone presence to be taken into account separately, the interaction term (\*) allows DBH to have a different effect on cone presence depending on the conifer species in question.

```r
# How does the likelihood of conifer cone presence change with tree age and species?

# Make a model with fixed and random effects 
dbh.mod <- glmer(Presence ~ DBH + Spec + (1 | Plot / ID) + (1 | Year), 
                 data = conesbi,
                 family = binomial(link = "logit"))
#summary(dbh.mod) 

# Make a model with interacting fixed effects 
dbh.mod.int <- glmer(Presence ~ DBH * Spec + (1 | Plot / ID) + (1 | Year), 
                     data = conesbi,
                     family = binomial(link = "logit"))
#summary(dbh.mod.int)
```

And finally, compare the models you've made. We'll extract the corrected Archaic Information Criterion value, AICc, from each model. 
-  The AICc is a number that indicates the model prediction error, so the lower the value the better the model. 
-  AICc puts a greater penalty on the number of parameters used in a model than AIC and is particularly good for smaller datasets. However, even with larger datasets like this, the AICc value ends up converging with AIC so it's generally best just to use AICc.
-  You can only compare AICc values on models using the same dataset and predicting the same response variable.

```r
# Compare ur models with AICc galfriend!
AICc(null.mod, dbh.mod, dbh.mod.int)
## species can't be included as a random effect because only 2 levels but seems like its better included as a fixed effect than not at all
```

This prints the AICc values of all the models with dbh.mod having the lowest. This shows that including the fixed effects individually gives us the strongest model.

<pre>


</pre>

<a name="section3"></a>

## 3.  Testing a logistic regression model

Now we have our model we can test its predictive capacity and check for outstanding outliers (assumption 4).

#### Cook's distance and outliers

We check for outliers using a Cook's distance plot. The Cook's distance of a datapoint indicates the effect on the model if it was deleted. Data points with particularly large Cook's distances therefore have disproportionate influences on the model and merit closer consideration. This can mean a) removing the point or replacing it with the mean value if there is an explanation for the discrepancy, or b) keeping the point in the model but making a careful note about this when reporting the regression results.

We'll create a simple general linear model (without the random effects so not a mixed GLM as Cook's distance can't take these into account). It is disputed about how far is too far with Cook's distance but generally 4/n (n being number of observations) or 0.05 is used. The function below should generate a plot in a new window that handily labels points that are possible outliers.

```r
# Check outliers with Cook's distance plot
model <- glm(Presence ~ DBH + Spec, family = "binomial", data = conesbi)
CookD(model) # this might pop up in another window
```
<p align="center">
   <img src="{{ site.baseurl }}/images/cooks_d.png" alt="drawing" width="80%">
</p>
<p align="center">
  Cooks distance plot with outliers labeled with reference numbers.
</p>

There are 3 possible outliers here. The index labels allow you to find them in the dataframe and check with metadata if there is any reason for removal. 

```r
conesbi[796,]
conesbi[817,]
conesbi[1082,]
```
These all appear to be pretty large old Engleman trees with no cones, but we have no scientific reason for removal.
Instead we'll keep them in the model and bear this in mind when reporting any results and conclusions.

#### Training and testing the model

By splitting up the dataset we can use some of the data to make the model and the remaining to test our model predictions against the true results.

Split up the data and rebuild our model using only the training dataset.

```r
# Test and train then model

# Make random sample reproducible (so you can come back to this and get the same set of numbers in the training and testing dataset)
set.seed(1) # you can use any integer here

# Use 70% of dataset as training set and remaining 30% as testing set
sample <- sample(c(TRUE, FALSE), nrow(conesbi), replace=TRUE, prob=c(0.7,0.3)) # randomly sample 70%
train <- conesbi[sample, ] # train dataset including 70% of the data
test <- conesbi[!sample, ] # make the test dataset with the remaining 30%

# Make a training model ysing the training data
train.mod <- glmer(Presence ~ DBH + Spec + (1 | Plot) + (1 | Year), # ID can't be included otherwise numbers wont be present for both training and testing datasets
                 data = train,
                 family = binomial)
```

Extract predictions made by the model for explanatory variable values from the test dataset.

```r
# Use your training model to make predictions for the test data
pdata <- predict(train_mod, newdata = test, type = "response")
```

We can then compare the models predictions for the test dataset to the real observations for the test data by building a confusion matrix.

```r
# Compute a confusion matrix comparing the predicted model outcomes to the real outcomes for the test dataset
confusionMatrix(data = as.factor(as.numeric(pdata>0.5)), reference = as.factor(test$Presence))  # the extracted predictions are probabilities and must be set as numerical 1/0 factors to match the original test data
```
<p align="center">
   <img src="{{ site.baseurl }}/images/conf_matrix.png" alt="drawing" width="100%">
</p>
<p align="center">
  Annotated confusion matrix report in Rstudio.
</p>

This shows the number of unmatching outcomes are low, showing the model is pretty strong. We can also quantify this by calculating the misclassification rate.

-   optimal : optimum cutoff probability of the model - the point on the explanatory variable at which it is more likely the binary response will be 1 than 0.
-   misclassification error rate : the proportion of observations (both positive and negative) that were not predicted correctly by the model

```r
# Calculate total misclassification error rate
optimal <- optimalCutoff(test$Presence, p_data())[1]  # find optimal cutoff probability for when 0 is predicted to 1 to use to maximize accuracy
misClassError(test$Presence, predicted, threshold=optimal)  # print misclas error rate
```

This prints the misclassification error rate as 23% which is good to bear in mind when writing your results and working with model predictions.

You can also use your model to make predictions given specific explanatory values. For example, to find the predictive probability of an Engleman spruce in Plot 1 with a DBH of 10 cm having cone presence in 2018 you could do the following:

```r
# Make predictions
new <- data.frame(DBH = 10, Spec  = "ABLA", Plot = "1", Year = "2018")  # create a new data frame defining predictor variables
predict(train.mod, new, type="response")  # print model predicted probability of it producing cones
```

This prints 0.1113638, meaning a tree with these attributes would have 11% likelihood of having seed cones.

<pre>


</pre>

<a name="section4"></a>

## 4.  Reporting and presenting model results

#### __Reporting results__

Now we know our model is pretty good we can draw any results from it. First we'll look at the model summary.

```r
# Look at the model outcomes
summary(dbh.mod)
```
<p align="center">
   <img src="{{ site.baseurl }}/images/mod_summary.png" alt="drawing" width="100%">
</p>
<p align="center">
  Annotated model summary report.
</p>

Everything is annotated here but all you'll probably use is the numbers in the second section down to compare potential models and the fixed effects summary for checking significance. However it's kind of out of fashion to report p-values. Instead we'll report the odds ratio and associated confidence intervals.

-   **Odds Ratio (OR) :** A measure of the odds of an event happening in one group compared to the odds of the same event happening in another group. Odds of an event is the likelihood that it occurs as a proportion of the likelihood it will not occur. For categorical variables each category is one group. For continuous variables one group represents a 1 unit increase in the variable from the other group.

    -   OR \> 1 : The event is more likely to occur
    -   OR \< 1 : The event is less likely to occur
    -   OR = 1 : The likelihood of the event stays the same

-   **Confidence Intervals (CI)** : Shows the error associated with the odds ratios estimates. If these extend so far as to cross 1 then the OR is not significant as there is a chance that the OR could be 1 meaning the variable has no impact on the likelihood of the event occuring.

-   **P-value** : A significance value taking into account the OR and the CI. This allows you to test your hypothesis .

You can print a table with these statistics using the code below and simply refer to it in your results statements if you're short on word space in your report. It can also be worth including an odds ratio diagram as the reader can quickly grasp if and how the fixed effects can predict the response.

```r
# Print a table for reporting values
tab_model(dbh.mod)
# Make fixed effects plots showing the odds ratio
plot_model(dbh.mod,
           type = "est", 
           show.values = TRUE)
```

<p align="center">
   <img src="{{ site.baseurl }}/images/CI_sumtab.png" alt="drawing" width="50%">
</p>
<p align="center">
  Summary reporting statistics (tab_model output).
</p>

<p align="center">
   <img src="{{ site.baseurl }}/images/odds_plot.png" alt="drawing" width="70%">
</p>
<p align="center">
  Odds ratio plot for final model fixed effects, DBH (tree diameter at breast height) and Spec (tree species, Engleman spruce in comparison to Subalpine fir). The odds ratio is represented by the red circle and 95% confidence interval by the horizontal red line. The odds ratio value is labeled on the plot and significance represented by asterisks.
</p>

We can then report :

-   A 1cm increase in tree DBH increases likelihood of cone presence by 16% (95% CI [.13, .19]; Figure 1). We can conclude tree size is a strong predictor of reproductive maturity.
-   Engleman spruce are 157% more likely to have seed cone presence than Subalpine fir (95% CI [.72, 2.84]; Figure 2).

(The Figures in the brackets refering to graphs that illustrate these results...)
<pre>

</pre>

#### __Presenting predictions__

We can now present our model predictions by plotting predicted values (marginal effects) for specific model terms. 
-  A marginal effect is the slope of the prediction function, measured for each variable in the model for each unit in the data. 
-  We will extract and plot predicted values in response to the model term DBH to show its relationship with cone presence. 
-  We will add a 95% confidence intervals to the plot showing the range of values that we expect the estimate to fall between 95% of the time. 

```r
# PLOT model predictions----

# Now we know the models are pretty good we can plot their prediction
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
           caption = '\n Figure 1: Probability of cone presence predicted from tree size for conifers in the \n southern Rocky Mountains, USA. ') + 
   plots_theme() +
   theme(plot.caption = element_text(size = 50, # for come reason this caption stuff just won't come through from the function we made so you have to reiterate it here
                                        hjust = 0))

# Save your plot
ggsave(filename = 'images/p_cone_dbh.png',
       width = 30,
       height = 15, 
       units = 'in')
```
<p align="center">
   <img src="{{ site.baseurl }}/images/p_cone_dbh.png" alt="drawing" width="70%">
</p>
<p align="center">
  Presenting a prediction distribution plot with 95% confidence intervals.
</p>

If your categorical predictor is significant you might want to show marginal effects for each separate category.

```r
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
```

<p align="center">
   <img src="{{ site.baseurl }}/images/p_cone_dbh_sp.png" alt="drawing" width="70%">
</p>
<p align="center">
  Presenting a prediction distribution plot with 95% confidence intervals.
</p>

<pre>


</pre>

## The End

This is the end of the tutorial. Taadaa the fog has lifted! Now you can have a walk outside, cup of tea and an unholy number of oreos then go get started on that report. Good stuff.

<p align="center">
   <img src="{{ site.baseurl }}/images/Spruce-habitat.png" alt="drawing" width="75%">
</p>
<p align="center">
  Engleman spruce forest (photo credit: <a href="http://nativeplantspnw.com/about-me/" target="_blank">D. Bressette</a>)
</p>


After this tutorial you should now be able to:

##### - check your data is appropriate for logistic regression

##### - fix data so it is appropriate for logistic regression

##### - build a logistic regression model

##### - test a logistic regression model

##### - interpret the results of your logistic regression model

##### - report and present the results and predictions of your model


Not sure if you understood anything you just read? Check yourself by doing the...
## CHALLENGE

For some conifers tree size has been shown to be a better indicator of cone production than tree age. This is possibly because tree size more strongly reflects a tree's access to resources than tree age.

See if this is true for this data by building a model using the tree age predictor variable instead of DBH. 
  - Is tree age a predictor of reproductive maturity for these confiers? 
  - Is it a stronger predictor than tree size? Does this differ between species?

<details>
  <summary>Click for code</summary>
 
   <pre>
   
   ```js

     # CHALLENGE----
      # CHECKING ASSUMPTIONS
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


      # Assumption 4
      # Check outliers of the explanatory variable
      boxplot(conesbi$Age, main = "Boxplot")

      # Assumption 5
      # Test linear explanatory~logit(response) relationship
      boxTidwell(Presence ~ Age, data = conesbi)

      # Assumption 6
      # Checking data quantity for explanatory variables
      obs <- length(conesbi$Presence) # find number of observstions
      summary(conesbi$Presence) # find least frequent outcome and its proportion.
      abs_lik <- (1 - (mean(conesbi$Presence))) # find likelihood of cone absence
      max_e_vars <- obs * abs_lik / 10
      max_e_vars # print the maximum number of explanatory variables allowed in the model

      # MAKING A MODEL
      # Boxplot to compare distributions of binomial distribution with continuous predictor
      data_dist <- melt(conesbi[, c("Presence", "Age")], # you can add in all your possible continuous predictor variables to have a look at the data distribution with it
                        id.vars="Presence") # set your response variable

      ggplot(data_dist, aes(factor(Presence), y = value, fill=factor(Presence))) +
        geom_boxplot() + 
        facet_wrap(~variable, scales="free_y") + # make panels of plots for each variable with a y scale that can be used for different units
        labs(caption = '\n Figure 1: Looking at data distribution of cone presence with tree age\n for conifers in the southern Rocky Mountains, USA ') + 
        plots_theme() +
        theme(plot.caption = element_text(size = 30,
                                          hjust = 0),
              legend.position = "right",
              legend.direction = "vertical",
              legend.title =  element_text(size = 30),
              legend.text = element_text(size = 30))

      # Binomial distribution with categorical on continuous predictor
      ggplot(conesbi, aes(x = Age, y = Presence, colour = Spec)) + 
        geom_point(size = 10, alpha = .2) + # make data points transparent so they can be seen overlayed
        labs(caption = '\n Figure 2: Looking at data distribution of cone presence with Age\n for Engelmann spruce (PIEN, Picea engelmannii) and subalpine fir (ABLA, Abies lasiocarpa) in\n the southern Rocky Mountains, USA ') + 
        plots_theme() +
        theme(plot.caption = element_text(size = 30,
                                          hjust = 0),
              legend.position = "right",
              legend.direction = "vertical",
              legend.title =  element_text(size = 30),
              legend.text = element_text(size = 20))

      # Null - make a null model to compare the others to

      null.mod <- glm(Presence ~ 1,  # there are no explanatory variables
                      data = conesbi,
                      family = binomial(link = "logit")) # setting the family as a binomial distribution but with a logit link to make it linear


      #summary(null.mod)  # you can print a summary of your models if you'd like and compare their outcomes


      # How does the likelihood of conifer cone presence change with tree age and species?

      # Make a model with fixed and random effects 
      age.mod <- glmer(Presence ~ Age + Spec + (1 | Plot / ID) + (1 | Year), 
                       data = conesbi,
                       family = binomial(link = "logit"))
      #summary(dbh.mod) 

      # Make a model with interacting fixed effects 
      age.mod.int <- glmer(Presence ~ Age * Spec + (1 | Plot / ID) + (1 | Year), 
                           data = conesbi,
                           family = binomial(link = "logit"))
      #summary(dbh.mod.int)

      # Compare ur models with AICc galfriend!
      AICc(null.mod, age.mod, age.mod.int)
      ## species can't be included as a random effect because only 2 levels but seems like its better included as a fixed effect than not at all



      # TESTING
      # Check outliers with Cook's distance plot
      model <- glm(Presence ~ Age + Spec, family = "binomial", data = conesbi)
      CookD(model) # this might pop up in another window

      conesbi[796,]
      conesbi[817,]
      conesbi[1082,]

      # Test and train then model

      # Make random sample reproducible (so you can come back to this and get the same set of numbers in the training and testing dataset)
      set.seed(2) # you can use any integer here

      # Use 70% of dataset as training set and remaining 30% as testing set
      sample <- sample(c(TRUE, FALSE), nrow(conesbi), replace=TRUE, prob=c(0.7,0.3)) # randomly sample 70%
      train <- conesbi[sample, ] # train dataset including 70% of the data
      test <- conesbi[!sample, ] # make the test dataset with the remaining 30%

      # PRESENT RESULTS
      # Look at the model outcomes
      summary(age.mod)
      # Print a table for reporting values
      tab_model(age.mod)
      # Make fixed effects plots showing the odds ratio
      plot_model(age.mod,
                 type = "est", 
                 show.values = TRUE)


      # Now we know the models are pretty good we can plot their prediction

      # Extract predictions
      predicted_age_sp <- plot_model(age.mod, type = "pred", terms = c("Age [all]", "Spec"))$data # extract model predictions in response to DBH and species

      # Make your marginal effects plots
      predicted_age_sp_plot <- ggplot(data = predicted_age_sp) + 
        geom_line(aes(x = x, # add a line showing the model's prediction
                      y = predicted, 
                      col = group_col), # colour the lines for each species
                  linewidth = 1) + 
        geom_ribbon(aes(x = x, # add a line to show 95% confidence intervals
                        ymin = conf.low,
                        ymax = conf.high,
                        fill = group_col), 
                    alpha = 0.3) + 
        labs(x = 'Tree Age (years)', # personalise labels
             y = 'Probability of cone presence',
             caption = '\n Figure 2: Probability of cone presence predicted from tree age for two conifer species\n in the southern Rocky Mountains, USA (+- 95% CI). ',
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

     ```
	</pre>
   
   
</details>



<hr>

<hr>

#### If you have any questions about completing this tutorial, please contact us on [ourcodingclub\@gmail.com](mailto:ourcodingclub@gmail.com){.email}

#### <a href="INSERT_SURVEY_LINK" target="_blank">We would love to hear your feedback on the tutorial, whether you did it in the classroom or online!</a>

<ul class="social-icons">

<li>

<h3>

<a href="https://twitter.com/our_codingclub" target="_blank"> Follow our coding adventures on Twitter! <i class="fa fa-twitter"></i></a>

</h3>

</li>

</ul>

###   Subscribe to our mailing list:

::: container
    <div class="block">
        <!-- subscribe form start -->
        <div class="form-group">
            <form action="https://getsimpleform.com/messages?form_api_token=de1ba2f2f947822946fb6e835437ec78" method="post">
            <div class="form-group">
                <input type='text' class="form-control" name='Email' placeholder="Email" required/>
            </div>
            <div>
                            <button class="btn btn-default" type='submit'>Subscribe</button>
                        </div>
                    </form>
        </div>
    </div>
:::
