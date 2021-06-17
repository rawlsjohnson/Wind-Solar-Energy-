# Predicting Solar and Wind Energy
Using Time Series Forecasting to predict the next 14 days of solar and wind energy production in Germany.  

Here is a link to a short presentation describing my project and the process.
https://www.loom.com/share/8c6088e02c374e4aa53d427113dc3d2d
## Project Summary
----------------------------------------------------------------------------------------------------------------------------------------
### Premise
The goal of my capstone project is to see if I can use machine learning and time series analysis to predict solar and wind energy output. The project allowed me to perform important Data Science processes including data cleaning, exploratory analysis, machine learning modeling, and evaluation. Being able to accurately forecast wind and solar power is very important in terms of improving the power grid’s efficiency. 


### Data Collection

I received my data from <https://open-power-system-data.org/> which is a free open source platform with data on power systems for 37 european countries. The website gave a drop down filtering system where you can download a CSV based on the time period you wanted to look at. The data file contained 15 variables about solar and wind by the hour however, I chose to focus on two features, solar generation actual and wind generation actual in MegaWatts (MW). These two features are what I wanted to predict. The period of time I chose to look at was from 2015 to 2020.

I chose to focus on a specific country, Germany, due to having the highest proportion of renewable energy than any other country and because it is a good indicator of where the rest of the world is headed.

### Data Cleaning & EDA
I began by reading the data file, parsing the dates and setting them as DatetimeIndex. My first steps I decided were to deal with all the missing values in the dataset. I found out that out of the 50401 entries, I had 104 solar generation nulls and 75 wind generation nulls. After exploring where they were and seeing that some days were missing. I chose to replace the nulls with the data from the previous day in order to keep the seasonality. I decided to average the generations over all the days in order to get better results. This would significantly reduce the noise and the number of data points in my dataframe. 

I decided to split the data frame into two in order to perform time series machine learning models. I performed separate EDA procedures on the two different data frames including looking at distributions and means. I also looked at Seasonal Decomposition of the time series looking at the trend and seasonal component. One thing I noticed is that there seems to be an increase in solar and wind production over the years and this is clearly due to the fact that Germany has been adding more and more solar and wind farms.

### Modeling Procedure
- Seasonal Decomposition
- Stationary Check
- ADF -test, KPSS test
- Autocorrelation and Partial Autocorrelation Plots 
- Train -Test split (13%)
- ARIMA 
- Facebook Prophet
- Used MAE and RMSE as a measure of explaining model

The first model I tried was the ARIMA model on the Solar generation data. I first had to check for stationary using ADF test and KPSS test. The series was not stationary due to a p-value greater than 0.05 however looking at the KPSS test, I found that that the series was stationary around a deterministic trend. This led me to believe that the ARIMA model was the right fight but will most likely have to use the differencing order to model. I took a look at the auto and partial autocorrelation plot in order to identify the order of the ARIMA model. They were very hard to tell but it looked like 2 for p and 3 for q. I did some additional  residual difference tests and they gave me different values for the order (p,q). I decided that I would try both and see what the results are. I split the data into test and train sets. I then decided to fit the arima model with two different orders for p and q and 1 for d.  I fit the model over 14 days and was able to get it. I ended up using some of the code from class to find the best order using mean absolute error. I thought the mean absolute error was a good indication because it showed how many MW that I was off by on average. This was the best indicator of a good model in my opinion. I found that the arima order of (2,1,5) produced the best MAE of 387.9 and a RMSE(root mean squared error) of 497.3.  These numbers are no way great if you consider accurately being able to use this in practice. 
	 
I decided that using Facebook Prophet would be the next best model to try.  I put the data in the correct format and performed a basic Facebook Prophet model to get a good sense of where to go. The prediction was able to get a good shape of the data however the MAE and RMSE were nothing to be impressed about. It appears that there was a lot of under fitting going on. I then did some model tuning by changing the growth to logistic and taking the cubed root of the y value.  I made a few other changes and kept changing them one by one until I got the smallest MAE of 333.3 and RMSE of 415. These results are slightly better than the ARIMA but if you consider 333 MW off, that is equal to powering a small city.  I tried adjusting many other parameters of the Facebook model and could not seem to achieve any better results. The graph of the predictions seem to look like they follow the general outline of the original data and you could produce some insights into the seasonal trend of actual solar generation and get a good sense of the high season and low. 
 
I performed the same steps (ARIMA / Prophet) above on the wind data however, due to the fact that wind power is much noisier. I was unable to get meaningful results. The Facebook model after tuning received a mean absolute error score of 7103, which in my opinion is very off even though the graph looks like it predicts it decently well. 


### Reflections 

Based on my findings, I believe I was able to produce results that could lead to something interesting, however, not that significant. I think my models are at a good starting place and more hypermater optimization could be performed in order to decrease the mean absolute error. When focusing on an entire country's energy supply, it seems a little too broad in order to get significant results. When your data on average is 333 MegaWatts off on average, that could be the difference of a small city. I think this process needs to be done on a much smaller scale like a particular wind farm. My goal was to predict the next 14 days and I did that but not to a degree that would make any real world impact. I think the next step would be to use Recurrent Neural Networks on the hour base data with weather features. This could give us more real world applicable results, however we would need to find a better data source.

### Some possible next steps:
- Add in extra regressors to Facebook Prophet model
- Recurrent Neural Networks using LSTM
- Try predicting the hours ahead instead of days
- Find better data for a more localized location like a specific wind farm or solar farm
- Hyperparameter optimization to choose the best parameters for various models
- Explore weather patterns and influence on Solar and Wind


## File Reference List

`time_series_60min_singleindex_filtered (3)` 
-	Contains all the solar and wind power data used for the project.

`Solar and Wind Energy Prediction.ipynb` 
-	Final notebook of all my steps(CLEANING,EDA,MODELING)

