Online News Popularity Analysis
================
Matthieu Cartron and Sneha Karanjai
2022-12-08

- <a href="#introduction" id="toc-introduction">1 Introduction</a>
- <a href="#setup" id="toc-setup">2 Setup</a>
- <a href="#data-import" id="toc-data-import">3 Data Import</a>
- <a href="#exploratory-data-analysis"
  id="toc-exploratory-data-analysis">4 Exploratory Data Analysis</a>
  - <a href="#column-description" id="toc-column-description">4.1 Column
    Description</a>
  - <a href="#summary-statistics" id="toc-summary-statistics">4.2 Summary
    Statistics</a>
  - <a href="#target-variable-distribution"
    id="toc-target-variable-distribution">4.3 Target Variable
    Distribution</a>
  - <a href="#title-tokens-vs-shares" id="toc-title-tokens-vs-shares">4.4
    Title Tokens vs Shares</a>
  - <a href="#number-of-links-in-the-articles-vs-shares"
    id="toc-number-of-links-in-the-articles-vs-shares">4.5 Number of Links
    in the Articles vs Shares</a>
  - <a href="#number-of-images-vs-shares"
    id="toc-number-of-images-vs-shares">4.6 Number of Images vs Shares</a>
  - <a href="#number-of-videos-vs-shares"
    id="toc-number-of-videos-vs-shares">4.7 Number of Videos vs Shares</a>
  - <a href="#days-of-the-week-and-shares"
    id="toc-days-of-the-week-and-shares">4.8 Days of the Week and Shares</a>
  - <a href="#title-polarity-vs-shares"
    id="toc-title-polarity-vs-shares">4.9 Title Polarity vs Shares</a>
  - <a href="#global-polarity-vs-shares"
    id="toc-global-polarity-vs-shares">4.10 Global Polarity vs Shares</a>
  - <a href="#subjectivity-and-shares" id="toc-subjectivity-and-shares">4.11
    Subjectivity and Shares</a>
  - <a
    href="#how-does-the-rate-of-negative-words-in-an-article-affect-the-shares"
    id="toc-how-does-the-rate-of-negative-words-in-an-article-affect-the-shares">4.12
    How does the rate of negative words in an article affect the Shares?</a>
  - <a href="#correlation-analysis" id="toc-correlation-analysis">4.13
    Correlation Analysis</a>
- <a href="#data-splitting" id="toc-data-splitting">5 Data Splitting</a>
- <a href="#modeling" id="toc-modeling">6 Modeling</a>
  - <a href="#linear-regression" id="toc-linear-regression">6.1 Linear
    Regression</a>
    - <a href="#linear-regression-with-dimensionality-reduction"
      id="toc-linear-regression-with-dimensionality-reduction">6.1.1 Linear
      Regression with Dimensionality Reduction</a>
  - <a href="#random-forest" id="toc-random-forest">6.2 Random Forest</a>
  - <a href="#boosted-tree" id="toc-boosted-tree">6.3 Boosted Tree</a>
- <a href="#model-comparison" id="toc-model-comparison">7 Model
  Comparison</a>
  - <a href="#train-model-comparison" id="toc-train-model-comparison">7.1
    Train Model Comparison</a>
  - <a href="#test-model-comparison" id="toc-test-model-comparison">7.2 Test
    Model Comparison</a>
- <a href="#conclusion" id="toc-conclusion">8 Conclusion</a>

# 1 Introduction

The following analysis uses the “Online News Popularity” data set from
the UCI machine learning repository. It consists of a number variables
describing different features of articles, each of which belonging to
one of six “channels.” These channels are effectively genres, and are
the following:

- Lifestyle
- Entertainment
- Business
- Social Media
- World News

For this analysis, we are primarily concerned with the “shares”
variable, which simply describes the number of times an article has been
shared. We often hear that news travels more quickly depending on its
content, title, and maybe even the number of images it uses. In a
similar vein, we would like, for each of the different data channels, to
use certain variables describing the articles to predict the number of
times an article might be shared. But how do we know which variables to
choose?

We could use simple intuition to pick variables. For example, it makes
sense to think that articles with high sentiment polarity (positive or
negative) would tend to, on average, have more shares. We could go
through the variables and pick those that we think would have the
greatest impact on the number of shares. The issue, however, is that
they may change from one data channel to the next. Are lifestyle
articles and world news articles going to be affected by the same
variables? If we choose the same variables across all the different data
channels, then this is the assumption we will be making. To avoid making
this assumption, we will automate the process of variable selection by
deleting one variable of each of the pairs of collinear variables.

# 2 Setup

``` r
library(tidyverse)
library(caret)
library(brainGraph)
library(corrplot)
library(GGally)
```

# 3 Data Import

``` r
unzippedNewDataCSV <- unzip("OnlineNewsPopularity.zip")

newsDataName <- read_csv(unzippedNewDataCSV[1]) # This is the names file
newsData <- read_csv(unzippedNewDataCSV[2])

head(newsData)
```

    ## # A tibble: 6 × 61
    ##   url            timedelta n_tokens_title n_tokens_content n_unique_tokens n_non_stop_words n_non_stop_uniqu… num_hrefs
    ##   <chr>              <dbl>          <dbl>            <dbl>           <dbl>            <dbl>             <dbl>     <dbl>
    ## 1 http://mashab…       731             12              219           0.664             1.00             0.815         4
    ## 2 http://mashab…       731              9              255           0.605             1.00             0.792         3
    ## 3 http://mashab…       731              9              211           0.575             1.00             0.664         3
    ## 4 http://mashab…       731              9              531           0.504             1.00             0.666         9
    ## 5 http://mashab…       731             13             1072           0.416             1.00             0.541        19
    ## 6 http://mashab…       731             10              370           0.560             1.00             0.698         2
    ## # … with 53 more variables: num_self_hrefs <dbl>, num_imgs <dbl>, num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_lifestyle <dbl>, data_channel_is_entertainment <dbl>,
    ## #   data_channel_is_bus <dbl>, data_channel_is_socmed <dbl>, data_channel_is_tech <dbl>, data_channel_is_world <dbl>,
    ## #   kw_min_min <dbl>, kw_max_min <dbl>, kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>,
    ## #   kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>, self_reference_min_shares <dbl>,
    ## #   self_reference_max_shares <dbl>, self_reference_avg_sharess <dbl>, weekday_is_monday <dbl>,
    ## #   weekday_is_tuesday <dbl>, weekday_is_wednesday <dbl>, weekday_is_thursday <dbl>, weekday_is_friday <dbl>, …

We will subset the data according to the channel passed to analyze
articles in one data channel at a time. Additionally according to the
data report, `url` and `timedelta` are two non-predictive columns so we
will remove them.

``` r
subsettingData <- function(data, area){
  #getting the naming convention as per the dataframe
  subsetVar <- paste("data_channel_is_", area, sep = "")
  
  # filtering the data and removing the data_channel_is_ columns, url, and timedelta
  subsetData <- data %>% 
    filter(!!as.symbol(subsetVar)==1) %>% 
    select(-c(starts_with("data_channel_is_"), url, timedelta))
  
  return(list(subsetData, subsetVar))
}

subsettingDataReturn <- subsettingData(newsData, params$channel)
data <- subsettingDataReturn[[1]]
channel <- subsettingDataReturn[[2]]
```

# 4 Exploratory Data Analysis

## 4.1 Column Description

Let us take a look at the columns available.

``` r
colnames(data)
```

    ##  [1] "n_tokens_title"               "n_tokens_content"             "n_unique_tokens"             
    ##  [4] "n_non_stop_words"             "n_non_stop_unique_tokens"     "num_hrefs"                   
    ##  [7] "num_self_hrefs"               "num_imgs"                     "num_videos"                  
    ## [10] "average_token_length"         "num_keywords"                 "kw_min_min"                  
    ## [13] "kw_max_min"                   "kw_avg_min"                   "kw_min_max"                  
    ## [16] "kw_max_max"                   "kw_avg_max"                   "kw_min_avg"                  
    ## [19] "kw_max_avg"                   "kw_avg_avg"                   "self_reference_min_shares"   
    ## [22] "self_reference_max_shares"    "self_reference_avg_sharess"   "weekday_is_monday"           
    ## [25] "weekday_is_tuesday"           "weekday_is_wednesday"         "weekday_is_thursday"         
    ## [28] "weekday_is_friday"            "weekday_is_saturday"          "weekday_is_sunday"           
    ## [31] "is_weekend"                   "LDA_00"                       "LDA_01"                      
    ## [34] "LDA_02"                       "LDA_03"                       "LDA_04"                      
    ## [37] "global_subjectivity"          "global_sentiment_polarity"    "global_rate_positive_words"  
    ## [40] "global_rate_negative_words"   "rate_positive_words"          "rate_negative_words"         
    ## [43] "avg_positive_polarity"        "min_positive_polarity"        "max_positive_polarity"       
    ## [46] "avg_negative_polarity"        "min_negative_polarity"        "max_negative_polarity"       
    ## [49] "title_subjectivity"           "title_sentiment_polarity"     "abs_title_subjectivity"      
    ## [52] "abs_title_sentiment_polarity" "shares"

``` r
str(data)
```

    ## tibble [7,346 × 53] (S3: tbl_df/tbl/data.frame)
    ##  $ n_tokens_title              : num [1:7346] 13 10 12 11 8 13 11 8 8 12 ...
    ##  $ n_tokens_content            : num [1:7346] 1072 370 989 97 1207 ...
    ##  $ n_unique_tokens             : num [1:7346] 0.416 0.56 0.434 0.67 0.411 ...
    ##  $ n_non_stop_words            : num [1:7346] 1 1 1 1 1 ...
    ##  $ n_non_stop_unique_tokens    : num [1:7346] 0.541 0.698 0.572 0.837 0.549 ...
    ##  $ num_hrefs                   : num [1:7346] 19 2 20 2 24 21 20 5 5 22 ...
    ##  $ num_self_hrefs              : num [1:7346] 19 2 20 0 24 19 20 2 3 22 ...
    ##  $ num_imgs                    : num [1:7346] 20 0 20 0 42 20 20 1 1 28 ...
    ##  $ num_videos                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ average_token_length        : num [1:7346] 4.68 4.36 4.62 4.86 4.72 ...
    ##  $ num_keywords                : num [1:7346] 7 9 9 7 8 10 7 10 9 9 ...
    ##  $ kw_min_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_min                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_max                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_min_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_max_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ kw_avg_avg                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ self_reference_min_shares   : num [1:7346] 545 8500 545 0 545 545 545 924 2500 545 ...
    ##  $ self_reference_max_shares   : num [1:7346] 16000 8500 16000 0 16000 16000 16000 924 2500 16000 ...
    ##  $ self_reference_avg_sharess  : num [1:7346] 3151 8500 3151 0 2830 ...
    ##  $ weekday_is_monday           : num [1:7346] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weekday_is_tuesday          : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday_is_wednesday        : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday_is_thursday         : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday_is_friday           : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday_is_saturday         : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weekday_is_sunday           : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ is_weekend                  : num [1:7346] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ LDA_00                      : num [1:7346] 0.0286 0.0222 0.0222 0.4583 0.025 ...
    ##  $ LDA_01                      : num [1:7346] 0.0288 0.3067 0.1507 0.029 0.0252 ...
    ##  $ LDA_02                      : num [1:7346] 0.0286 0.0222 0.2434 0.0287 0.025 ...
    ##  $ LDA_03                      : num [1:7346] 0.0286 0.0222 0.0222 0.0297 0.025 ...
    ##  $ LDA_04                      : num [1:7346] 0.885 0.627 0.561 0.454 0.9 ...
    ##  $ global_subjectivity         : num [1:7346] 0.514 0.437 0.543 0.539 0.539 ...
    ##  $ global_sentiment_polarity   : num [1:7346] 0.281 0.0712 0.2986 0.1611 0.2883 ...
    ##  $ global_rate_positive_words  : num [1:7346] 0.0746 0.0297 0.0839 0.0309 0.0696 ...
    ##  $ global_rate_negative_words  : num [1:7346] 0.0121 0.027 0.0152 0.0206 0.0116 ...
    ##  $ rate_positive_words         : num [1:7346] 0.86 0.524 0.847 0.6 0.857 ...
    ##  $ rate_negative_words         : num [1:7346] 0.14 0.476 0.153 0.4 0.143 ...
    ##  $ avg_positive_polarity       : num [1:7346] 0.411 0.351 0.428 0.567 0.427 ...
    ##  $ min_positive_polarity       : num [1:7346] 0.0333 0.1364 0.1 0.4 0.1 ...
    ##  $ max_positive_polarity       : num [1:7346] 1 0.6 1 0.8 1 1 1 0.35 1 1 ...
    ##  $ avg_negative_polarity       : num [1:7346] -0.22 -0.195 -0.243 -0.125 -0.227 ...
    ##  $ min_negative_polarity       : num [1:7346] -0.5 -0.4 -0.5 -0.125 -0.5 -0.5 -0.5 -0.2 -0.5 -0.5 ...
    ##  $ max_negative_polarity       : num [1:7346] -0.05 -0.1 -0.05 -0.125 -0.05 -0.05 -0.05 -0.05 -0.125 -0.05 ...
    ##  $ title_subjectivity          : num [1:7346] 0.455 0.643 1 0.125 0.5 ...
    ##  $ title_sentiment_polarity    : num [1:7346] 0.136 0.214 0.5 0 0 ...
    ##  $ abs_title_subjectivity      : num [1:7346] 0.0455 0.1429 0.5 0.375 0 ...
    ##  $ abs_title_sentiment_polarity: num [1:7346] 0.136 0.214 0.5 0 0 ...
    ##  $ shares                      : num [1:7346] 505 855 891 3600 17100 2800 445 783 1500 1800 ...

Whew! That is a long list of columns to analyze. Instead of analyzing
them all, let us think about our data. What might we expect to be
related to how many times an article is shared? We hear frequently about
how the dissemination of news and the content thereof are related in
some way. For our data channels, let’s pay close attention to the number
of shares (dissemination) and variables that we might be able to link to
it. Can we find any interesting relationships in the exploratory
analysis? And do these relationships change across the different
channels? Maybe the sharing of lifestyle articles is less correlated
with sentiment than, say, world news articles.

First, let’s take a look at the variable descriptions for some better
understanding. Here is a data description from the UCI Machine Learning
Repository:

- n_tokens_title: Number of words in the title
- n_tokens_content Number of words in the content
- n_unique_tokens: Rate of unique words in the content
- n_non_stop_unique_tokens: Rate of unique non-stop words in the content
- num_hrefs: Number of links
- num_self_hrefs: Number of links to other articles published by
  Mashable
- num_imgs: Number of images
- num_videos: Number of videos
- average_token_length: Average length of the words in the content
- num_keywords: Number of keywords in the metadata
- self_reference_min_shares: Min. shares of referenced articles in
  Mashable
- self_reference_max_shares: Max. shares of referenced articles in
  Mashable
- self_reference_avg_sharess: Avg. shares of referenced articles in
  Mashable
- global_subjectivity: Text subjectivity
- global_sentiment_polarity: Text sentiment polarity
- global_rate_positive_words: Rate of positive words in the content
- global_rate_negative_words: Rate of negative words in the content
- rate_positive_words: Rate of positive words among non-neutral tokens
- rate_negative_words: Rate of negative words among non-neutral tokens
- title_subjectivity: Title subjectivity
- title_sentiment_polarity: Title polarity
- abs_title_subjectivity: Absolute subjectivity level
- abs_title_sentiment_polarity: Absolute polarity level
- shares: Number of shares (target)

Below we run the five-number summary for each of the variables thus far
still included.

## 4.2 Summary Statistics

``` r
print(paste("******Summary Statistics of", channel, "******"))
```

    ## [1] "******Summary Statistics of data_channel_is_tech ******"

``` r
summary(data)
```

    ##  n_tokens_title  n_tokens_content n_unique_tokens  n_non_stop_words n_non_stop_unique_tokens   num_hrefs      
    ##  Min.   : 4.00   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000           Min.   :  0.000  
    ##  1st Qu.: 9.00   1st Qu.: 256.0   1st Qu.:0.4598   1st Qu.:1.0000   1st Qu.:0.6165           1st Qu.:  5.000  
    ##  Median :10.00   Median : 405.0   Median :0.5336   Median :1.0000   Median :0.6897           Median :  7.000  
    ##  Mean   :10.19   Mean   : 571.6   Mean   :0.5310   Mean   :0.9971   Mean   :0.6829           Mean   :  9.417  
    ##  3rd Qu.:12.00   3rd Qu.: 728.0   3rd Qu.:0.6017   3rd Qu.:1.0000   3rd Qu.:0.7568           3rd Qu.: 11.000  
    ##  Max.   :20.00   Max.   :5530.0   Max.   :0.9143   Max.   :1.0000   Max.   :1.0000           Max.   :120.000  
    ##  num_self_hrefs       num_imgs        num_videos      average_token_length  num_keywords      kw_min_min    
    ##  Min.   :  0.000   Min.   : 0.000   Min.   : 0.0000   Min.   :0.000        Min.   : 2.000   Min.   : -1.00  
    ##  1st Qu.:  2.000   1st Qu.: 1.000   1st Qu.: 0.0000   1st Qu.:4.424        1st Qu.: 7.000   1st Qu.: -1.00  
    ##  Median :  3.000   Median : 1.000   Median : 0.0000   Median :4.592        Median : 8.000   Median :  0.00  
    ##  Mean   :  4.641   Mean   : 4.435   Mean   : 0.4472   Mean   :4.582        Mean   : 7.776   Mean   : 29.98  
    ##  3rd Qu.:  6.000   3rd Qu.: 6.000   3rd Qu.: 1.0000   3rd Qu.:4.756        3rd Qu.: 9.000   3rd Qu.:  4.00  
    ##  Max.   :116.000   Max.   :65.000   Max.   :73.0000   Max.   :5.857        Max.   :10.000   Max.   :294.00  
    ##    kw_max_min       kw_avg_min        kw_min_max         kw_max_max       kw_avg_max       kw_min_avg    
    ##  Min.   :     0   Min.   :   -1.0   Min.   :     0.0   Min.   :     0   Min.   :     0   Min.   :   0.0  
    ##  1st Qu.:   483   1st Qu.:  162.8   1st Qu.:     0.0   1st Qu.:690400   1st Qu.:139970   1st Qu.:   0.0  
    ##  Median :   729   Median :  254.1   Median :   990.5   Median :843300   Median :186859   Median : 830.5  
    ##  Mean   :  1155   Mean   :  317.7   Mean   :  7010.4   Mean   :737045   Mean   :213482   Mean   : 996.6  
    ##  3rd Qu.:  1100   3rd Qu.:  374.2   3rd Qu.:  5600.0   3rd Qu.:843300   3rd Qu.:287317   3rd Qu.:1965.5  
    ##  Max.   :104100   Max.   :14716.9   Max.   :663600.0   Max.   :843300   Max.   :753450   Max.   :3347.3  
    ##    kw_max_avg       kw_avg_avg    self_reference_min_shares self_reference_max_shares self_reference_avg_sharess
    ##  Min.   :     0   Min.   :    0   Min.   :     0.0          Min.   :     0            Min.   :     0            
    ##  1st Qu.:  3444   1st Qu.: 2346   1st Qu.:   767.2          1st Qu.:  1700            1st Qu.:  1469            
    ##  Median :  3739   Median : 2698   Median :  1300.0          Median :  4000            Median :  2775            
    ##  Mean   :  4584   Mean   : 2746   Mean   :  4423.7          Mean   : 11787            Mean   :  7219            
    ##  3rd Qu.:  4700   3rd Qu.: 3079   3rd Qu.:  2700.0          3rd Qu.:  8400            3rd Qu.:  5250            
    ##  Max.   :104100   Max.   :19429   Max.   :663600.0          Max.   :663600            Max.   :663600            
    ##  weekday_is_monday weekday_is_tuesday weekday_is_wednesday weekday_is_thursday weekday_is_friday weekday_is_saturday
    ##  Min.   :0.0000    Min.   :0.0000     Min.   :0.0000       Min.   :0.0000      Min.   :0.0000    Min.   :0.00000    
    ##  1st Qu.:0.0000    1st Qu.:0.0000     1st Qu.:0.0000       1st Qu.:0.0000      1st Qu.:0.0000    1st Qu.:0.00000    
    ##  Median :0.0000    Median :0.0000     Median :0.0000       Median :0.0000      Median :0.0000    Median :0.00000    
    ##  Mean   :0.1681    Mean   :0.2007     Mean   :0.1929       Mean   :0.1783      Mean   :0.1346    Mean   :0.07147    
    ##  3rd Qu.:0.0000    3rd Qu.:0.0000     3rd Qu.:0.0000       3rd Qu.:0.0000      3rd Qu.:0.0000    3rd Qu.:0.00000    
    ##  Max.   :1.0000    Max.   :1.0000     Max.   :1.0000       Max.   :1.0000      Max.   :1.0000    Max.   :1.00000    
    ##  weekday_is_sunday   is_weekend         LDA_00            LDA_01            LDA_02            LDA_03       
    ##  Min.   :0.00000   Min.   :0.0000   Min.   :0.01819   Min.   :0.01819   Min.   :0.01818   Min.   :0.01818  
    ##  1st Qu.:0.00000   1st Qu.:0.0000   1st Qu.:0.02232   1st Qu.:0.02225   1st Qu.:0.02500   1st Qu.:0.02224  
    ##  Median :0.00000   Median :0.0000   Median :0.02860   Median :0.02857   Median :0.03333   Median :0.02857  
    ##  Mean   :0.05391   Mean   :0.1254   Mean   :0.07409   Mean   :0.06523   Mean   :0.11018   Mean   :0.06186  
    ##  3rd Qu.:0.00000   3rd Qu.:0.0000   3rd Qu.:0.04007   3rd Qu.:0.04000   3rd Qu.:0.15772   3rd Qu.:0.04000  
    ##  Max.   :1.00000   Max.   :1.0000   Max.   :0.83882   Max.   :0.72201   Max.   :0.82213   Max.   :0.74564  
    ##      LDA_04        global_subjectivity global_sentiment_polarity global_rate_positive_words global_rate_negative_words
    ##  Min.   :0.01846   Min.   :0.0000      Min.   :-0.33912          Min.   :0.00000            Min.   :0.000000          
    ##  1st Qu.:0.54575   1st Qu.:0.4141      1st Qu.: 0.09713          1st Qu.:0.03303            1st Qu.:0.008696          
    ##  Median :0.73332   Median :0.4598      Median : 0.14379          Median :0.04187            Median :0.013632          
    ##  Mean   :0.68864   Mean   :0.4568      Mean   : 0.14577          Mean   :0.04290            Mean   :0.014436          
    ##  3rd Qu.:0.86658   3rd Qu.:0.5037      3rd Qu.: 0.19102          3rd Qu.:0.05179            3rd Qu.:0.018987          
    ##  Max.   :0.92719   Max.   :0.8127      Max.   : 0.65000          Max.   :0.15217            Max.   :0.086168          
    ##  rate_positive_words rate_negative_words avg_positive_polarity min_positive_polarity max_positive_polarity
    ##  Min.   :0.0000      Min.   :0.0000      Min.   :0.0000        Min.   :0.00000       Min.   :0.0000       
    ##  1st Qu.:0.6667      1st Qu.:0.1667      1st Qu.:0.3139        1st Qu.:0.05000       1st Qu.:0.6000       
    ##  Median :0.7524      Median :0.2454      Median :0.3560        Median :0.10000       Median :0.8000       
    ##  Mean   :0.7466      Mean   :0.2506      Mean   :0.3571        Mean   :0.09917       Mean   :0.7719       
    ##  3rd Qu.:0.8333      3rd Qu.:0.3304      3rd Qu.:0.4013        3rd Qu.:0.10000       3rd Qu.:1.0000       
    ##  Max.   :1.0000      Max.   :1.0000      Max.   :1.0000        Max.   :1.00000       Max.   :1.0000       
    ##  avg_negative_polarity min_negative_polarity max_negative_polarity title_subjectivity title_sentiment_polarity
    ##  Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.0000       Min.   :0.00000    Min.   :-1.00000        
    ##  1st Qu.:-0.2850       1st Qu.:-0.6000       1st Qu.:-0.1250       1st Qu.:0.00000    1st Qu.: 0.00000        
    ##  Median :-0.2220       Median :-0.4000       Median :-0.1000       Median :0.06667    Median : 0.00000        
    ##  Mean   :-0.2288       Mean   :-0.4513       Mean   :-0.1027       Mean   :0.25310    Mean   : 0.08636        
    ##  3rd Qu.:-0.1667       3rd Qu.:-0.2500       3rd Qu.:-0.0500       3rd Qu.:0.47708    3rd Qu.: 0.15000        
    ##  Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.0000       Max.   :1.00000    Max.   : 1.00000        
    ##  abs_title_subjectivity abs_title_sentiment_polarity     shares      
    ##  Min.   :0.0000         Min.   :0.000                Min.   :    36  
    ##  1st Qu.:0.1556         1st Qu.:0.000                1st Qu.:  1100  
    ##  Median :0.5000         Median :0.000                Median :  1700  
    ##  Mean   :0.3447         Mean   :0.136                Mean   :  3072  
    ##  3rd Qu.:0.5000         3rd Qu.:0.200                3rd Qu.:  3000  
    ##  Max.   :0.5000         Max.   :1.000                Max.   :663600

## 4.3 Target Variable Distribution

Let’s take a look at the distribution of our target variable using a
histogram.

``` r
ggplot(data) +
  aes(x = shares) +
  geom_histogram(bins = 26L, fill = "#112446") +
  labs(title = "Distribution of Shares") +
  theme_gray()
```

![](tech_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

*What does the distribution show? Does the distribution follow a known
distribution? Is there skewness? What might these features tell us about
the number of shares? The number of shares is plotted along the x-axis,
with frequency (count) on the y-axis.*

## 4.4 Title Tokens vs Shares

Now let’s analyze the affect of the different variables on the shares.
Starting with the number of words in the title and how they affect the
shares.

``` r
data %>% 
  group_by(n_tokens_title) %>% 
  summarise(avgShares = mean(shares)) %>% 
  ggplot() +
  aes(x = avgShares, y = n_tokens_title) +
  geom_point(shape = "circle", size = 1.5, colour = "#112446") +
  labs(title = "Average Shares vs Title Tokens") +
  theme_gray()
```

![](tech_files/figure-gfm/unnamed-chunk-3-1.png)<!-- --> *The average
number of shares is plotted on the x-axis while the numvber of words in
the article title is plotted on the y-axis. Can we see any relationship
between the two variables?*

## 4.5 Number of Links in the Articles vs Shares

``` r
data %>% 
  group_by(num_hrefs) %>% 
  summarise(avgShares = mean(shares)) %>% 
  ggplot() +
  aes(x = avgShares, y = num_hrefs) +
  geom_point(shape = "circle", size = 1.5, colour = "#112446") +
  labs(title = "Average Shares vs Number of Links") +
  theme_gray()
```

![](tech_files/figure-gfm/unnamed-chunk-4-1.png)<!-- --> *The average
number of shares is plotted on the x-axis while the number of hyperlinks
is plotted on the y-axis. Like with the previous plot, we use a scatter
plot because we have two numeric variables, with the average number of
shares being continuous. Can we see any relationship between the two
variables?*

## 4.6 Number of Images vs Shares

``` r
data %>% 
  group_by(factor(num_imgs)) %>% 
  summarise(sumShares = sum(shares)) %>% 
  ggplot() +
  aes(x = `factor(num_imgs)`, y = sumShares) +
  geom_col(fill = "#112446") +
  labs(title = "Shares vs Images", x = "Number of Images", y = "Shares(Sum)") +
  theme_minimal()
```

![](tech_files/figure-gfm/unnamed-chunk-5-1.png)<!-- --> *The above bar
plot demonstrates the relationship between the number of images in an
article (x-axis) and the sum of the shares the article experienced. Can
we see any patterns in the above visualization?*

## 4.7 Number of Videos vs Shares

``` r
data %>% 
  group_by(factor(num_videos)) %>% 
  summarise(sumShares = sum(shares)) %>% 
  ggplot() +
  aes(x = `factor(num_videos)`, y = sumShares) +
  geom_col(fill = "#112446") +
  labs(title = "Shares vs Videos", x = "Number of Videos", y = "Shares(Sum)") +
  theme_minimal()
```

![](tech_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

*In the above bar plot the number of videos featured in an article is
plotted against the summed shares per video number. Do we notice any
patterns? Can we make any comparisons between this plot (with videos) vs
the previous plot, which looks at the number of images in an article?*

## 4.8 Days of the Week and Shares

``` r
mon <- data %>% 
  select(starts_with("weekday_is_monday"), shares) %>% 
  group_by(weekday_is_monday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_monday) %>% 
  filter(day==1) 
mon$day[mon$day==1] <- "MON"

tue <- data %>% 
  select(starts_with("weekday_is_tuesday"), shares) %>% 
  group_by(weekday_is_tuesday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_tuesday) %>% 
  filter(day==1)
tue$day[tue$day==1] <- "TUE"


wed <- data %>% 
  select(starts_with("weekday_is_wednesday"), shares) %>% 
  group_by(weekday_is_wednesday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_wednesday) %>% 
  filter(day==1)
wed$day[wed$day==1] <- "WED"


thu <- data %>% 
  select(starts_with("weekday_is_thursday"), shares) %>% 
  group_by(weekday_is_thursday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_thursday) %>% 
  filter(day==1)
thu$day[thu$day==1] <- "THU"

fri <- data %>% 
  select(starts_with("weekday_is_friday"), shares) %>% 
  group_by(weekday_is_friday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_friday) %>% 
  filter(day==1)
fri$day[fri$day==1] <- "FRI"

sat <- data %>% 
  select(starts_with("weekday_is_saturday"), shares) %>% 
  group_by(weekday_is_saturday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_saturday) %>% 
  filter(day==1)
sat$day[sat$day==1] <- "SAT"

sun <- data %>% 
  select(starts_with("weekday_is_sunday"), shares) %>% 
  group_by(weekday_is_sunday) %>% 
  summarise(sumShares = sum(shares)) %>% 
  rename(day = weekday_is_sunday) %>% 
  filter(day==1)
sun$day[sun$day==1] <- "SUN"

mon %>% 
  bind_rows(tue, wed, thu, fri, sat, sun) %>% 
  ggplot() +
  aes(x = day, y = sumShares) +
  geom_col(fill = "#112446") +
  labs(title = "Most Shared Articles by the Day of the Week") +
  theme_gray()
```

![](tech_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

*The above bar plot looks at the sum of shares given for each day of the
week. Are there any patterns? Are there differences in the number of
shares between weekdays and the weekend? If so, what might cause this?
Are articles also most likely to be published on certain days of the
week, and thus more likely to be shared on those days? We can
speculate.*

*In the three scatter plots below, we take a magnifying glass to some of
the variables measuring features of article sentiment. We suspect there
might be some patterns below (not guaranteed!). Can we use this as a
starting point for investigating how article sentiment influences the
dissemination of information (if at all)?*

## 4.9 Title Polarity vs Shares

Polarity is a float which lies in the range of \[-1,1\] where 1 refers
to a positive statement and -1 refers to a negative statement. Does
title polarity affect the average number of shares?

``` r
data %>% 
  ggplot() +
  aes(x = title_sentiment_polarity, y = shares) +
  geom_point() +
  geom_jitter() +
  labs(title = "Shares vs Title Polarity", x = "Title Polarity", y = "Number of Shares") +
  theme_minimal()
```

![](tech_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

*The above scatter plot looks at title polarity (how negative or
positive an article title might be) and the number of shares for a given
article. Can we see any initial patterns worth exploring? *

## 4.10 Global Polarity vs Shares

``` r
data %>% 
  ggplot() +
  aes(x = global_sentiment_polarity, y = shares) +
  geom_point() +
  geom_jitter() +
  labs(title = "Shares vs Text Polarity", x = "Text Polarity", y = "Number of Shares") +
  theme_minimal()
```

![](tech_files/figure-gfm/unnamed-chunk-9-1.png)<!-- -->

*The above scatter plot is similar to the previous scatter plot, though
this time we take a look at the text polarity (how positive or negative
the words of the article are) and plot it against the number of times a
given article is shared (y-axis). Again, do we notice any patterns?*

## 4.11 Subjectivity and Shares

Subjective sentences generally refer to personal opinion, emotion or
judgment whereas objective refers to factual information. Subjectivity
is a float which lies in the range of \[0,1\]. A value closer to 0 means
an opinion or an emotion and 1 means a fact. How does the text having a
factual tone or an author’s emotion/opinion affect the total shares?

``` r
ggplot(data) +
  aes(x = shares, y = global_subjectivity) +
  geom_point(shape = "circle", size = 1.5, colour = "#112446") +
  labs(title = "Shares vs Text Subjectivity", x = "Text Subjectivity", y = "Number of Shares") +
  theme_minimal()
```

![](tech_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

*In the above scatter plot, we plot the text subjectivity against the
number of times an article is shared (y-axis). Though subjectivity is
not sentiment, we might have reason to suspect that they could be
related–are subjective articles more mysterious, more enticing, more
prone to “clickbait”? Does this scatter plot seem to convey anything
like this?*

## 4.12 How does the rate of negative words in an article affect the Shares?

``` r
data %>% 
  group_by(rate_negative_words) %>% 
  summarise(avgShares = mean(shares)) %>% 
  ggplot() +
  aes(x = avgShares, y = rate_negative_words) +
  geom_point(shape = "circle", size = 1.5, colour = "#112446") +
  labs(title = "Average Shares vs Rate of Negative Words") +
  theme_gray()
```

![](tech_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

*Here we see how the rate of the usage of negative words throughout the
article tends to affect the shares. Does an article with higher number
of negative words tend to have lesser shares?*

## 4.13 Correlation Analysis

Now that we have completed analysis of how the shares changes with the
different variables, we do notice that there are way too many variables
in this dataset. Feeding all these variables into the training models
would mean “Garbage In and Garbage Out”. One of the easiest ways to
choose the variables to fit into the models is by checking the
correlation. Potential predictors with high correlation between each
other can prove problematic as they introduce multicollinearity into the
model. We can remove some of this redundancy from the outset. While
there are some models that thrive on correlated predictors. other models
may benefit from reducing the level of correlation between the
predictors.

Let us first understand the pair plots for all the variables explaining
keywords.

``` r
pairs(~ kw_min_min + kw_max_min + kw_min_max + kw_avg_max + kw_max_avg + kw_avg_min + kw_max_max + kw_min_avg + kw_avg_avg, data = data)
```

![](tech_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

``` r
cor(data[, c('kw_min_min', 'kw_max_min', 'kw_min_max', 'kw_avg_max', 'kw_max_avg', 'kw_avg_min', 'kw_max_max', 'kw_min_avg', 'kw_avg_avg')])
```

    ##             kw_min_min   kw_max_min   kw_min_max  kw_avg_max   kw_max_avg  kw_avg_min  kw_max_max   kw_min_avg
    ## kw_min_min  1.00000000  0.032556210 -0.061969786 -0.52844004 -0.060805198  0.19484197 -0.82803972 -0.163607566
    ## kw_max_min  0.03255621  1.000000000 -0.049016953 -0.06860305  0.643350514  0.93274391 -0.01799662 -0.006835493
    ## kw_min_max -0.06196979 -0.049016953  1.000000000  0.31994949  0.006775132 -0.09169619  0.06990947  0.309160807
    ## kw_avg_max -0.52844004 -0.068603052  0.319949492  1.00000000  0.077663805 -0.23250915  0.60478143  0.358339486
    ## kw_max_avg -0.06080520  0.643350514  0.006775132  0.07766380  1.000000000  0.57286326  0.09530493  0.077213801
    ## kw_avg_min  0.19484197  0.932743910 -0.091696191 -0.23250915  0.572863264  1.00000000 -0.17367964 -0.062286543
    ## kw_max_max -0.82803972 -0.017996621  0.069909468  0.60478143  0.095304926 -0.17367964  1.00000000  0.191667476
    ## kw_min_avg -0.16360757 -0.006835493  0.309160807  0.35833949  0.077213801 -0.06228654  0.19166748  1.000000000
    ## kw_avg_avg -0.21756780  0.396903549  0.160564151  0.36591883  0.752188233  0.32680430  0.31752350  0.528480069
    ##            kw_avg_avg
    ## kw_min_min -0.2175678
    ## kw_max_min  0.3969035
    ## kw_min_max  0.1605642
    ## kw_avg_max  0.3659188
    ## kw_max_avg  0.7521882
    ## kw_avg_min  0.3268043
    ## kw_max_max  0.3175235
    ## kw_min_avg  0.5284801
    ## kw_avg_avg  1.0000000

``` r
kwCorData <- as.data.frame(as.table(cor(data[, c('kw_min_min', 'kw_max_min', 'kw_min_max', 'kw_avg_max', 'kw_max_avg', 'kw_avg_min', 'kw_max_max', 'kw_min_avg', 'kw_avg_avg')])))

colRemove <- kwCorData %>% 
  filter(abs(Freq)>0.8 & Freq!=1 )

colRemove <- as.vector(colRemove$Var2)

data <- data %>% 
  select(-all_of(colRemove))
```

This removes all the highly correlated keyword variables that convey the
same information. Now we will similarly investigate the self-referenced
shares.

``` r
pairs(~ self_reference_avg_sharess + self_reference_max_shares + self_reference_min_shares, data = data)
```

![](tech_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

If we find that any of the self_reference shares variables have a
correlation of greater than 0.8 with one another, then we will eliminate
it from the analysis. Again, this is done to limit the multicollinearity
in the models we build below as well as reduce their dimension. We want
to simplify our models from the outset as much as possible without
losing predictors that will explain much of the variability in the
number of times an article is shared.

``` r
srCorData <- as.data.frame(as.table(cor(data[, c('self_reference_avg_sharess', 'self_reference_max_shares', 'self_reference_min_shares')])))

colRemove <- srCorData %>% 
  filter(abs(Freq)>0.8 & Freq!=1)

colRemove <- as.vector(colRemove$Var2)

data <- data %>% 
  select(-all_of(colRemove))
```

In this next step, we examine our remaining variables to see if any
share a correlation of 0.70 or higher. If so, we will remove it from the
data.

``` r
descrCor <- cor(data) 
highlyCorVar <- findCorrelation(descrCor, cutoff = .85)
data <- data[,-highlyCorVar]
```

Again, we do not want to remove both the highly correlated variables.
For example, if we were looking at the variables temperature in
Farenheit and temperature in Celcius in predicting the number of people
at a beach, both variable would be telling us the same thing, but we
would still want to keep one of them because of its probable importance
to the model. We will also remove `is_weekend` from our analysis as the
variables `weekday_is_sunday` and `weekday_is_saturday` capture the same
information.

``` r
data <- data %>% 
  select(-c("is_weekend"))
```

We are now down from 61 columns to 43 columns.

Let us finally do a correlation plot for all variables with threshold
greater than 0.55 for the the present dataframe.

``` r
cols <- names(data)

corrDf <- data.frame(t(combn(cols,2)), stringsAsFactors = F) %>%
  rowwise() %>%
  mutate(v = cor(data[,X1], data[,X2]))

corrDf <- corrDf %>% 
  filter(abs(v)>0.55) %>% 
  arrange(desc(v))
corrDf
```

    ## # A tibble: 13 × 3
    ## # Rowwise: 
    ##    X1                         X2                            v[,1]
    ##    <chr>                      <chr>                         <dbl>
    ##  1 global_rate_negative_words rate_negative_words           0.811
    ##  2 kw_max_avg                 kw_avg_avg                    0.752
    ##  3 avg_negative_polarity      min_negative_polarity         0.731
    ##  4 n_non_stop_words           average_token_length          0.700
    ##  5 title_subjectivity         abs_title_sentiment_polarity  0.688
    ##  6 title_sentiment_polarity   abs_title_sentiment_polarity  0.623
    ##  7 avg_positive_polarity      max_positive_polarity         0.565
    ##  8 avg_negative_polarity      max_negative_polarity         0.559
    ##  9 num_hrefs                  num_self_hrefs                0.554
    ## 10 title_subjectivity         abs_title_subjectivity       -0.566
    ## 11 LDA_02                     LDA_04                       -0.581
    ## 12 n_tokens_content           n_non_stop_unique_tokens     -0.615
    ## 13 global_sentiment_polarity  rate_negative_words          -0.709

``` r
#turn corr back into matrix in order to plot with corrplot
correlationMat <- reshape2::acast(corrDf, X1~X2, value.var="v")
  
#plot correlations visually
corrplot(correlationMat, is.corr=FALSE, tl.col="black", na.label=" ")
```

![](tech_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

Now that we have a more concise data set, let’s zero in on the
relationship between our target variable, shares, and the remaining
variables. Below we extract the five variables that have the highest
correlation with the shares variable. This may be a valuable insight
prior to training our models.

``` r
sharesCor <- cor(data[ , colnames(data) != "shares"],  # Calculate correlations
                data$shares)

sharesCor <- data.frame(sharesCor)
sharesCor$names <- rownames(sharesCor)
rownames(sharesCor) <- NULL

sharesCor <- sharesCor %>% 
  rename(corrcoeff=sharesCor) %>% 
  arrange(desc(abs(corrcoeff))) %>% 
  head(6)

par(mfrow=c(2,3))
for (i in sharesCor$names) {
  plot(data$shares, data[[i]], ylab = i, main = paste("Shares vs", i))
}
```

![](tech_files/figure-gfm/unnamed-chunk-20-1.png)<!-- -->

``` r
#for (i in sharesCor$names) {
 # ggpairs(data[[i]], ylab = i, main = paste("Shares vs", i))
#}
```

# 5 Data Splitting

Data splitting is an important aspect of data science, particularly for
creating predictive models based on data. This technique helps ensure
the creation of data models and processes that use data models – such as
machine learning – are accurate. In a basic two-part data split, the
training data set is used to train and fit models. Training sets are
commonly used to estimate different parameters or to compare different
model performance. The testing data set is used after the training is
done; we see if our trained models are effective in predicting future
values. We will use a 70-30 split on the dataset.

``` r
train_index <- createDataPartition(data$shares, p = 0.7, 
                                   list = FALSE)
train <- data[train_index, ]
test <- data[-train_index, ]
```

We will check the shape of the train and test set

``` r
print("The train set dimensions")
```

    ## [1] "The train set dimensions"

``` r
dim(train)
```

    ## [1] 5145   43

``` r
print("The test set dimensions")
```

    ## [1] "The test set dimensions"

``` r
dim(test)
```

    ## [1] 2201   43

# 6 Modeling

We will be comparing linear and ensemble techniques for predicting
shares. Each section below elucidates the model used and the reasoning
behind it.

## 6.1 Linear Regression

A simple linear regression refers to a linear equation that captures the
relationship between a response variable, $Y$, and a predictor variable
$X$. The relationship is modeled below:

$$Y = \beta_0 + \beta_1X_1 +\epsilon i$$

Where $\beta_0$ is the intercept and $\beta_1$ is the slope of the line.
This relationship can be extended to the case in which the response
variable is modeled as a function of more than one predictor variable.
This is the case of a multiple linear regression, which is as follows:

$$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + … + \beta_nX_n + \epsilon i$$

Where $\beta_0$ is the intercept and all $\beta$ are slope coefficients.
For both simple and multiple linear regression cases, the Method of
Least Squares is widely used in summarizing the data. The least squares
method minimizes values of $\beta_0$ and all $\beta_n$, seen below:

$$\sum_{i = 1}^{n} (yi - \beta_0 - \sum_{j = 1}^{k} \beta_j x_{ij}^2)$$

Since we are dealing with 43 variables, it is probably important to know
that we would need to employ a feature selection/dimension reduction
technique. Feature selection is the process of reducing the number of
input variables when developing a predictive model. It is desirable to
reduce the number of input variables to both reduce the computational
cost of modeling and, in some cases, to improve the performance of the
model. To prove this, we will first fit a full-model (with all the
available variables) with multiple linear regression.

``` r
trControl <- trainControl(method = "repeatedcv", number = 5, repeats = 5)

mlrWithoutVS <- train(shares ~ .,
                     data = train,
                     preProcess = c("center", "scale"),
                     method = "lm", 
                     trControl = trControl)

summary(mlrWithoutVS)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8882  -1770  -1064    117  99594 
    ## 
    ## Coefficients: (2 not defined because of singularities)
    ##                              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                  3010.846     66.865  45.028  < 2e-16 ***
    ## n_tokens_title                -26.011     70.417  -0.369 0.711853    
    ## n_tokens_content              166.325    115.349   1.442 0.149383    
    ## n_non_stop_words              -79.679    103.008  -0.774 0.439248    
    ## n_non_stop_unique_tokens      128.409    105.797   1.214 0.224907    
    ## num_hrefs                     307.455     95.643   3.215 0.001314 ** 
    ## num_self_hrefs               -333.182     87.626  -3.802 0.000145 ***
    ## num_imgs                       74.887     88.767   0.844 0.398910    
    ## num_videos                    144.903     68.298   2.122 0.033916 *  
    ## average_token_length         -131.313     96.604  -1.359 0.174111    
    ## num_keywords                  -96.537     74.899  -1.289 0.197491    
    ## kw_min_max                    -46.922     73.299  -0.640 0.522112    
    ## kw_avg_max                   -264.466     82.888  -3.191 0.001428 ** 
    ## kw_min_avg                   -249.340     99.960  -2.494 0.012649 *  
    ## kw_max_avg                   -533.458    133.917  -3.983 6.89e-05 ***
    ## kw_avg_avg                    999.352    161.778   6.177 7.03e-10 ***
    ## weekday_is_monday            -408.356    128.946  -3.167 0.001550 ** 
    ## weekday_is_tuesday           -399.430    134.483  -2.970 0.002991 ** 
    ## weekday_is_wednesday         -326.951    132.686  -2.464 0.013769 *  
    ## weekday_is_thursday          -458.925    129.224  -3.551 0.000387 ***
    ## weekday_is_friday            -265.941    118.925  -2.236 0.025381 *  
    ## weekday_is_saturday           -91.353    100.552  -0.909 0.363646    
    ## weekday_is_sunday                  NA         NA      NA       NA    
    ## LDA_00                         -8.753     69.137  -0.127 0.899262    
    ## LDA_01                        -72.866     69.184  -1.053 0.292290    
    ## LDA_02                         91.173     71.868   1.269 0.204637    
    ## LDA_03                          4.631     69.531   0.067 0.946902    
    ## LDA_04                             NA         NA      NA       NA    
    ## global_subjectivity           148.179     82.924   1.787 0.074009 .  
    ## global_sentiment_polarity     -93.450    174.739  -0.535 0.592812    
    ## global_rate_positive_words   -245.821    131.170  -1.874 0.060978 .  
    ## global_rate_negative_words     76.943    180.035   0.427 0.669121    
    ## rate_negative_words           -94.978    201.730  -0.471 0.637792    
    ## avg_positive_polarity         -55.123    134.575  -0.410 0.682112    
    ## min_positive_polarity         -69.810     88.090  -0.792 0.428113    
    ## max_positive_polarity         147.079    103.685   1.419 0.156101    
    ## avg_negative_polarity        -199.585    174.917  -1.141 0.253913    
    ## min_negative_polarity         144.452    153.300   0.942 0.346091    
    ## max_negative_polarity         262.317    115.525   2.271 0.023210 *  
    ## title_subjectivity            -52.823    101.832  -0.519 0.603971    
    ## title_sentiment_polarity      121.435     88.296   1.375 0.169094    
    ## abs_title_subjectivity         73.901     85.057   0.869 0.384979    
    ## abs_title_sentiment_polarity  114.589    113.120   1.013 0.311116    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4796 on 5104 degrees of freedom
    ## Multiple R-squared:  0.03393,    Adjusted R-squared:  0.02636 
    ## F-statistic: 4.481 on 40 and 5104 DF,  p-value: < 2.2e-16

``` r
mlrWithoutVS$results
```

    ##   intercept     RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1      TRUE 4767.179 0.02044057 2224.479 772.7065 0.00791291 100.4463

As we can see, the Root Mean Square Error of the model is
`mlrWithoutVS$results$RMSE`. Now let us see how this changes with the
Principle Components Analysis below.

### 6.1.1 Linear Regression with Dimensionality Reduction

Principle Components Analysis (PCA) is a dimension-reduction technique
that can be extended to regression. In a PCA we find linear combinations
of the predictor variables that account for most of the variability in
the model. What this does is it reduces the number of variables $p$ into
$m$ principal components, allowing for a reduction in complexity all the
while retaining most of the variability of the $p$ variables. We extend
this to regression by treating our $m$ principle components as
predictors, though we cannot interpret them in the same way.

``` r
pcs <- prcomp(select(train, -c("shares")), scale = TRUE, center = TRUE)
summary(pcs)
```

    ## Importance of components:
    ##                            PC1     PC2     PC3     PC4     PC5     PC6    PC7     PC8     PC9    PC10    PC11    PC12
    ## Standard deviation     2.00805 1.78649 1.63326 1.56736 1.48757 1.35269 1.2798 1.21517 1.19236 1.13699 1.11853 1.10700
    ## Proportion of Variance 0.09601 0.07599 0.06351 0.05849 0.05269 0.04357 0.0390 0.03516 0.03385 0.03078 0.02979 0.02918
    ## Cumulative Proportion  0.09601 0.17200 0.23551 0.29400 0.34669 0.39025 0.4293 0.46440 0.49826 0.52904 0.55882 0.58800
    ##                           PC13    PC14    PC15    PC16    PC17    PC18    PC19    PC20   PC21    PC22    PC23    PC24
    ## Standard deviation     1.09186 1.07530 1.06697 1.06017 1.03412 1.02649 1.02636 1.00706 0.9590 0.88991 0.84980 0.83095
    ## Proportion of Variance 0.02838 0.02753 0.02711 0.02676 0.02546 0.02509 0.02508 0.02415 0.0219 0.01886 0.01719 0.01644
    ## Cumulative Proportion  0.61639 0.64392 0.67102 0.69778 0.72325 0.74833 0.77341 0.79756 0.8195 0.83832 0.85551 0.87195
    ##                           PC25    PC26    PC27    PC28    PC29    PC30    PC31   PC32    PC33    PC34    PC35    PC36
    ## Standard deviation     0.81424 0.77913 0.77193 0.72973 0.68486 0.66646 0.62728 0.5869 0.54858 0.50854 0.47896 0.45749
    ## Proportion of Variance 0.01579 0.01445 0.01419 0.01268 0.01117 0.01058 0.00937 0.0082 0.00717 0.00616 0.00546 0.00498
    ## Cumulative Proportion  0.88773 0.90219 0.91638 0.92905 0.94022 0.95080 0.96017 0.9684 0.97553 0.98169 0.98715 0.99214
    ##                           PC37    PC38    PC39    PC40      PC41      PC42
    ## Standard deviation     0.31990 0.30910 0.27092 0.24301 1.315e-12 2.761e-15
    ## Proportion of Variance 0.00244 0.00227 0.00175 0.00141 0.000e+00 0.000e+00
    ## Cumulative Proportion  0.99457 0.99685 0.99859 1.00000 1.000e+00 1.000e+00

How many principle components should we use? This is somewhat
subjective. Consider the plot below. How many principle components would
be required in order to retain say 80 or 90 percent of the variability
in the data? If we can effectively reduce the number of variables by way
of this method, then we may want to consider a regression of these
principle components, even if we lose some interpretability.

``` r
par(mfrow = c(1, 2))
plot(pcs$sdev^2/sum(pcs$sdev^2), xlab = "Principal Component", 
         ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = 'b')
plot(cumsum(pcs$sdev^2/sum(pcs$sdev^2)), xlab = "Principal Component", 
ylab = "Cum. Prop of Variance Explained", ylim = c(0, 1), type = 'b')
```

![](tech_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

``` r
pcaVar <- as.vector(cumsum(pcs$sdev^2/sum(pcs$sdev^2)))
for (i in seq_along(pcaVar)) {
  if(pcaVar[i] > 0.9 & pcaVar[i] < 0.92){
    pcaIndex = i
  }
}

pc_train <- predict(pcs, train)
pc_train <- data.frame(pc_train)
pc_train <- pc_train %>% 
  select(all_of(c(1:pcaIndex))) %>% 
  mutate(shares = train$shares)
pc_test <- data.frame(predict(pcs, test))  %>% 
  mutate(shares = test$shares)
```

We will now fit a multiple linear regression using these principle
components.

``` r
mlrWitVS <- train(shares ~ .,
                  data = pc_train,
                  preProcess = c("center", "scale"),
                  method = "lm", 
                  trControl = trControl)

summary(mlrWitVS)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -9577  -1803  -1143     61 100104 
    ## 
    ## Coefficients:
    ##              Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) 3010.8461    67.1938  44.808  < 2e-16 ***
    ## PC1          253.7619    67.2004   3.776 0.000161 ***
    ## PC2          240.0627    67.2004   3.572 0.000357 ***
    ## PC3           77.0735    67.2004   1.147 0.251468    
    ## PC4         -194.5863    67.2004  -2.896 0.003800 ** 
    ## PC5         -150.0675    67.2004  -2.233 0.025583 *  
    ## PC6          174.8959    67.2004   2.603 0.009279 ** 
    ## PC7          -58.7820    67.2004  -0.875 0.381763    
    ## PC8         -222.3209    67.2004  -3.308 0.000945 ***
    ## PC9          -99.0719    67.2004  -1.474 0.140469    
    ## PC10         133.0157    67.2004   1.979 0.047826 *  
    ## PC11         -11.2629    67.2004  -0.168 0.866904    
    ## PC12         -75.5346    67.2004  -1.124 0.261057    
    ## PC13        -259.5121    67.2004  -3.862 0.000114 ***
    ## PC14          -0.7467    67.2004  -0.011 0.991135    
    ## PC15          33.4926    67.2004   0.498 0.618224    
    ## PC16          80.7439    67.2004   1.202 0.229598    
    ## PC17          17.3666    67.2004   0.258 0.796085    
    ## PC18          21.2474    67.2004   0.316 0.751879    
    ## PC19        -210.8454    67.2004  -3.138 0.001713 ** 
    ## PC20         120.6058    67.2004   1.795 0.072757 .  
    ## PC21        -140.4962    67.2004  -2.091 0.036604 *  
    ## PC22        -187.8015    67.2004  -2.795 0.005215 ** 
    ## PC23          96.8747    67.2004   1.442 0.149482    
    ## PC24         108.5371    67.2004   1.615 0.106345    
    ## PC25         -34.7860    67.2004  -0.518 0.604728    
    ## PC26         -16.9649    67.2004  -0.252 0.800701    
    ## PC27         -23.5922    67.2004  -0.351 0.725548    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 4820 on 5117 degrees of freedom
    ## Multiple R-squared:  0.02193,    Adjusted R-squared:  0.01677 
    ## F-statistic: 4.249 on 27 and 5117 DF,  p-value: 1.285e-12

``` r
mlrWitVS$results
```

    ##   intercept    RMSE   Rsquared      MAE   RMSESD RsquaredSD    MAESD
    ## 1      TRUE 4791.65 0.01335688 2237.286 697.0419 0.00606134 84.78323

Now that we have a Multiple Linear Regression with PCA, let us see how a
Lasso Regression on the original dataset competes in terms of variable
selection.

``` r
tuneGrid <- expand.grid(
  .fraction = seq(0, 1, by = 0.1)
)

lassoModel <- train(
  shares ~ .,
  data = train,
  method = 'lasso',
  preProcess = c("center", "scale"),
  trControl = trControl,
  tuneGrid = tuneGrid
)
```

``` r
lassoModel$results
```

    ##    fraction     RMSE   Rsquared      MAE   RMSESD  RsquaredSD    MAESD
    ## 1       0.0 4807.921        NaN 2272.463 730.3331          NA 88.24395
    ## 2       0.1 4778.607 0.01667446 2231.657 725.4820 0.008889538 94.74401
    ## 3       0.2 4773.579 0.01838556 2225.227 724.6558 0.009115371 92.89317
    ## 4       0.3 4770.670 0.01930333 2221.361 722.9502 0.009193178 91.45729
    ## 5       0.4 4768.701 0.02005816 2219.341 721.4297 0.009228586 90.81161
    ## 6       0.5 4767.039 0.02076665 2218.116 720.5781 0.009349812 90.57520
    ## 7       0.6 4766.387 0.02115014 2217.893 719.7692 0.009413247 90.49562
    ## 8       0.7 4766.674 0.02127785 2218.609 719.0588 0.009486393 90.49556
    ## 9       0.8 4767.456 0.02131008 2220.088 718.2619 0.009525452 90.63277
    ## 10      0.9 4768.373 0.02128890 2221.765 717.9425 0.009555086 91.14970
    ## 11      1.0 4769.515 0.02120406 2223.659 717.7832 0.009580624 91.88012

## 6.2 Random Forest

The random forest model refers to an ensemble method of either
classification or regression. In this case, we are predicting a
continuous response variable, and are thus using the latter case. The
random forest creates numerous trees from bootstrap samples of the data.
Bootstrap samples are simply samples taken from the data and are of the
same size (sample $n$ equals bootstrap $m$), meaning that an observation
from the sample data could be used twice in the bootstrap sample, for
example. A tree is fit to each bootstrap sample, and for each fit a
random subset (generally $m = p/3$ predictors) of predictors is chosen.
This is done with the tuning parameter mtry.

Generally speaking, random forests predict more accurately because the
results of the fitted trees are averaged across all trees. This
averaging reduces variance.

``` r
tuneGrid = expand.grid(mtry = 1:3)

rfModel <- train(shares ~ .,
                  data = train,
                  method = "rf", 
                  trControl = trControl,
                  tuneGrid = tuneGrid)
```

Looking at the Variable Importance Plot :

``` r
plot(varImp(rfModel))
```

![](tech_files/figure-gfm/unnamed-chunk-33-1.png)<!-- -->

``` r
rfModel$results
```

    ##   mtry     RMSE   Rsquared      MAE   RMSESD  RsquaredSD    MAESD
    ## 1    1 4760.819 0.02831224 2231.363 630.1847 0.009152838 93.92921
    ## 2    2 4761.853 0.02790989 2273.006 623.2966 0.010051047 90.85058
    ## 3    3 4774.182 0.02534980 2291.567 623.5242 0.009002800 93.15887

## 6.3 Boosted Tree

Boosted tree models, like the random forest, are an ensemble tree-based
method that can be used for classification or regression. Again, in our
case, we are predicting a continuous response and are using regression.

Unlike the random forest, in the boosted tree method, trees are grown
sequentially, and for each tree the residuals are treated as the
response. This is exactly true for the first tree. Updated predictions
can be modeled by the following:

$$\hat{y} = \hat{y}(x) + \lambda \hat{y}^b(x)$$

Below we fit our boosted tree model to the training data set.

``` r
tuneGrid = expand.grid(n.trees = seq(5, 30, 5), interaction.depth = seq(1,10,1), shrinkage = 0.1, n.minobsinnode = 20)

# fit the model
boostingModel <- train(shares ~ .,
                  data = train,
                  method = "gbm", 
                  trControl = trControl,
                  tuneGrid = tuneGrid,
                  verbose = FALSE)
```

# 7 Model Comparison

## 7.1 Train Model Comparison

Now although using the accuracy on the testing data is the gold standard
for model comparison, it can be imperative to check the train model
accuracy to see how the models have fit the dataset. If there is a huge
difference in train accuracies then we know that a certain model does
not fit the data well. Here is a summarisation of the models with their
hyperparameters and model metrics.

``` r
trainModelComparison <- mlrWithoutVS$results[which.min(mlrWithoutVS$results$RMSE),] %>% 
  bind_rows(mlrWitVS$results[which.min(mlrWitVS$results$RMSE),],
            lassoModel$results[which.min(lassoModel$results$RMSE),], 
            rfModel$results[which.min(mlrWithoutVS$results$RMSE),],
            boostingModel$results[which.min(boostingModel$results$RMSE),]) %>% 
  mutate(Model = c("MLR", "MLR with PCA", "Lasso", "Random Forest", "Boosted Tree with PCA")) %>% 
  select(Model, everything())
trainModelComparison
```

    ##                   Model intercept     RMSE   Rsquared      MAE   RMSESD  RsquaredSD     MAESD fraction mtry shrinkage
    ## 1                   MLR      TRUE 4767.179 0.02044057 2224.479 772.7065 0.007912910 100.44631       NA   NA        NA
    ## 2          MLR with PCA      TRUE 4791.650 0.01335688 2237.286 697.0419 0.006061340  84.78323       NA   NA        NA
    ## 3                 Lasso        NA 4766.387 0.02115014 2217.893 719.7692 0.009413247  90.49562      0.6   NA        NA
    ## 4         Random Forest        NA 4760.819 0.02831224 2231.363 630.1847 0.009152838  93.92921       NA    1        NA
    ## 5 Boosted Tree with PCA        NA 4741.952 0.02052051 2220.764 879.0542 0.009357069 123.83964       NA   NA       0.1
    ##   interaction.depth n.minobsinnode n.trees
    ## 1                NA             NA      NA
    ## 2                NA             NA      NA
    ## 3                NA             NA      NA
    ## 4                NA             NA      NA
    ## 5                 1             20      30

We see that the model with the lowest RMSE value is Boosted Tree with
PCA with an RMSE value of 4741.95. The model that performs the poorest
MLR with PCA with an RMSE value of 4791.65 which means that the model
was incapable of fitting the data well.

## 7.2 Test Model Comparison

Now that we have created our trained models (models fit to the training
data) we should now see how accurately they predict future values. Once
we have evaluated each of the models, we should be able to compare them
to see which is best at making predictions on future data. We can do
this by comparing the predicted values of the tested with the actual
test set values.

Withe function `postResample()` we can find our RMSE on the test data
set and compare it across models.

``` r
predLinearTest <- predict(mlrWithoutVS, test)
testMLR<- postResample(pred = predLinearTest, obs = test$shares)
```

``` r
predPCAtest <- predict(mlrWitVS, pc_test)
testPCA <- postResample(pred = predPCAtest, obs = pc_test$shares)
```

``` r
predLassoTest <- predict(lassoModel, test)
testLasso <- postResample(pred = predLassoTest, test$shares)
```

``` r
predRandomForest <- predict(rfModel, test)
testRandomForest <- postResample(pred = predRandomForest, obs = test$shares)
```

``` r
predBoosting <- predict(boostingModel, test)
testBoosting <- postResample(pred = predBoosting, obs = test$shares)
```

In comparing the above models, we should be looking at which among the
models best minimizes the RMSE, as the model with the lowest RMSE will,
on average, make the most accurate predictions.

``` r
testModelComparison <- testMLR %>% 
  bind_rows(testPCA, testLasso, testRandomForest, testBoosting) %>% 
  mutate(Model = c("MLR", "MLR with PCA", "Lasso", "Random Forest", "Boosted Tree with PCA")) %>% 
  select(Model, everything())
testModelComparison
```

    ## # A tibble: 5 × 4
    ##   Model                   RMSE Rsquared   MAE
    ##   <chr>                  <dbl>    <dbl> <dbl>
    ## 1 MLR                   14655.  0.00963 2436.
    ## 2 MLR with PCA          14690.  0.00378 2439.
    ## 3 Lasso                 14655.  0.00963 2436.
    ## 4 Random Forest         14672.  0.00872 2429.
    ## 5 Boosted Tree with PCA 14677.  0.00823 2416.

We see that the model with the lowest RMSE value is Lasso with an RMSE
value of 1.465509^{4}. The model that performs the poorest MLR with PCA
with an RMSE value of 1.468952^{4}.

# 8 Conclusion

Here’s is general observation on the modelling with data with high
dimension : When there are large number of features with less
data-sets(with low noise), linear regressions may outperform Decision
trees/random forests. There is no thumb rule on which model will perform
the best on what kind of dataset. If you are dealing with a dataset with
high dimensionality then your first approach must be to decrease this
dimensionality before fitting the models. Both LASSO and PCA are
dimensionality reduction techniques. Both methods can reduce the
dimensionality of a dataset but follow different styles. LASSO, as a
feature selection method, focuses on deletion of irrelevant or redundant
features. PCA, as a dimension reduction method, combines the features
into a smaller number of aggregated components (a.k.a., the new
features).
