# Online News Popularity Analysis

This dataset summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. The goal of this repository is to predict the number of shares in social networks, i.e. how popular any given article is. The dataset is publicly available at University of California Irvine Machine Learning Repository. 

Mashable Inc. is a digital media website founded in 2005. It has been described as a “one stop shop” for social media. As of November 2015, it has over 6,000,000 Twitter followers and over 3,200,000 fans on Facebook.

The methodology used was to first read in the data and filter on the channel dynamically. We then analyze the shares as the predictor for each channel. We conduct exploratory data analysis to understand the variables that are important to the target variable and then fit regression models like linear regression and ensemble techniques. Finally, these models are compared in terms of their model metrics to choose the best performing model for each channel to predict the sales. 

# Packages Used 

- tidyverse 
- caret 
- brainGraph 
- corrplot 
- ggplot2 

# Links to the Final Reports 

- [Lifestyle Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/lifestyle.html) 
- [Business Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/business.html) 
- [Social Media Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/socialmedia.html)
- [World Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/world.html)
- [Technology Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/technology.html) 
- [Entertainment Report :](https://mcartron10.github.io/Stat558_Project3_CartronKaranjai/entertainment.html) 

# Code to generate reports for each channel 

```
# import library
library(tidyverse)
library(rmarkdown)

#read in the data
unzippedNewDataCSV <- unzip("OnlineNewsPopularity.zip")
newsData <- read_csv(unzippedNewDataCSV[2])

#get the channel names
col_names <- c(colnames(newsData)) 
channel_colnames <- col_names[grepl("data_channel_is", col_names)]
channel_names <- gsub("^.*_", "", channel_colnames)

#create filenames
output_file <- paste0(channel_names, ".html")

#create a list for each team with just the team name parameter
params = lapply(channel_names, FUN = function(x){list(channel = x)})

#put into a data frame 
reports <- tibble(output_file, params)

## #need to use x[[1]] to get at elements since tibble doesn't simplify
apply(reports, MARGIN = 1,
      FUN = function(x){
				render(input = "Stat558_Project3_CartronKaranjai.Rmd", output_file =x[[1]], params = x[[2]])
 				})
```
