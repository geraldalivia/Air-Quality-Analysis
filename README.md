# Air Quality Analysis Dashboard

## Project Overview
This is a final project from Dicoding in the "Belajar Analisis Data Dengan Python" course to make analysis and create a dashboard.

This dashboard analyzes air pollution data from Dongsi and Wanliu stations from 2013 to 2017, focusing on PM10 by examining daily patterns and correlations with meteorological factors.

## Live Dashboard
[Streamlit-Geralda Livia](https://analysis-air-quality-geraldalivia.streamlit.app/)

## Data Source
Raw Dataset air quality from [Dicoding Air Quality](https://air-quality-geraldalivia.streamlit.app/). For the analysis and development dashboard use the cleaned data version from [Cleaned Data](https://drive.google.com/file/d/1--d07m7J4CniV6pfScx_S6sde5XJu05J/view?usp=drive_link). Analysis focus on PM10 levels and other meteorological related data.

## Research Questions
1. What is the daily pattern of PM10 concentrations at Dongsi and Wanliu stations for the period 2013-2017?
2. Is there a correlation between meteorological factors (TEMP, DEWP and PRES) and PM10 concentration levels at the two stations?        

## Installation and Setup
### Create virtual environment use pipeenv
   To install pipenv
   ```
   pip install pipenv
   ```
   To create virtual environment
   ```
   pipenv install
   ```
   To activate virtual environment
   ```
   pipenv shell
   ```
### Install the required packages
   The packages needed to run the analysis on both Colab and the dashboard
   ```
   pip install pandas numpy matplotlib gdown plotly seaborn streamlit
   ```
   or by the following command
   ```
   pip install -r requirements.txt
   ```
### Run the Dashboard 
   Navigate to the  `air-quality-dashboard.py` and runn the streamlit App
    ```
    streamlit run air-quality-dashboard.py
    ```
## Project Overview
1. Import Library
   - Streamlit
   - Pandas
   - Numpy
   - Matplotlib
   - Seaborn
   - Gdown
   - Plotly
3. Data Wrangling
   - Gathering Data
   - Accessing Data
   - Cleaning Data
5. Exploratory Data Analysis (EDA):
   - Daily Pattern
   - Correlation
   - Further Analysis
7. Conclusion

## About Dashboard
- **Overview**: Basic information and statistics about the dataset
- **Daily Patterns**: Visualization of hourly PM10 concentrations throughout the day
- **Correlation Analysis**: Relationship between PM10 and meteorological parameters
- **Further Analysis**: To understand the different factors contributing to PM10 variations over time.
- **Summary**: Project Conclusion

## About me
- **Name**: Geralda Livia Nugraha
- **Email Address**: mc299d5x1168@student.devacademy.id
- **Dicoding ID**: [MC299D5X1168](https://www.dicoding.com/users/alddar/)
