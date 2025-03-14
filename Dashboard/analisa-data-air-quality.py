import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# Title page
st.set_page_config(page_title=" Air Quality Analysis by Geralda Livia")

# Title of the dashboard
st.title('Data Analysis Project: Air Quality Dashboard')

# Description
st.write('This is a dashboard that show analyzes air pollution data from Dongsi and Wanliu stations from 2013-2017, focusing on PM10 concentrations and meteorological factors')

# About me
st.markdown("""
### About Me
- **Name**: Geralda Livia Nugraha
- **Email Address**: mc299d5x1168@student.devacademy.id
- **Dicoding ID**: [MC299D5X1168](https://www.dicoding.com/users/alddar/)

### Project Overview
This project aims to analyze air quality data from Dongsi and Wanliu stations in China from 2013 to 2017, focusing on PM10 concentrations by examining daily patterns and investigating correlations with meteorological factors. The insights gained will enhance the understanding of pollution patterns and their driving factors, contributing to more effective air quality management strategies. The findings can guide policy decisions on traffic management and emission controls, help residents plan outdoor activities to avoid peak pollution hours, provide scientific evidence for environmental protection measures, support urban planning that considers air quality factors, and contribute to public health initiatives by identifying high-risk pollution periods.

### Define Question  
1. What is the daily pattern of PM10 concentrations at Dongsi and Wanliu stations for the period 2013-2017?
2. Is there a correlation between meteorological factors (TEMP, DEWP and PRES) and PM10 concentration levels at the two stations (Dongsi and Wanliu)?        
""")


# Load Data using ID from GDrive
file_id = "1--d07m7J4CniV6pfScx_S6sde5XJu05J"
output = "data.csv"
url = f'https://drive.google.com/uc?id={file_id}'
# Download data
@st.cache_data
def load_data():
    # Download file
    gdown.download(url, output, quiet=False)
    data = pd.read_csv(output)
    return data
try:
    data = load_data()
    st.write(f"Data berhasil dimuat: {data.shape[0]} baris")
    st.dataframe(data.head())
except Exception as e:
    st.error(f"Error: {e}")

# Load dataset
#data = pd.read_csv(r"D:\ML Engineer DBS Foundation X Dicoding\latihan_python\all_data_air_quality.csv", encoding='utf-8', engine='python')


# Display raw data sample
with st.expander("Dataset Overview"):
    st.dataframe(data.head())
    st.write(f"Dataset shape: {data.shape}")
    
    # Display basic info
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Types:")
        st.write(data.dtypes)
    
    with col2:
        st.write("Missing Values:")
        missing_values = data.isnull().sum()
        missing_percentage = (data.isnull().sum() / len(data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_percentage.round(2)
        })
        st.dataframe(missing_df)


# Data preprocessing
@st.cache_data
def preprocess_data(df):
    # Create datetime column
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    
    # Extract time components
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour_of_day'] = df['datetime'].dt.hour
    df['month_name'] = df['datetime'].dt.month_name()
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    # Impute missing values
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_cols:
        if df[column].isnull().sum() > 0:
            df[column].fillna(df[column].median(), inplace=True)
    
    # Filter for stations of interest
    dongsi_data = df[df['station'] == 'Dongsi'].copy()
    wanliu_data = df[df['station'] == 'Wanliu'].copy()
    
    return df, dongsi_data, wanliu_data


# Preprocess data
data, dongsi_data, wanliu_data = preprocess_data(data)


# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", 
                         ["Overview", 
                          "Daily PM10 Patterns", 
                          "Meteorological Correlations", 
                          "Further Analysis",
                          "Summary"])

# Overview page
if page == "Overview":
    st.header("Dataset Overview")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.dataframe(data.describe())
    
    # Station information
    st.subheader("Station Information")
    station_counts = data['station'].value_counts()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Records per Station:")
        st.dataframe(station_counts)
    
    with col2:
        fig = px.pie(values=station_counts.values, names=station_counts.index, 
                     title="Data Distribution by Station")
        st.plotly_chart(fig)
    
    # Year and month distribution
    st.subheader("Temporal Distribution")
    
    year_counts = data['year'].value_counts().sort_index()
    month_counts = data['month'].value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=year_counts.index, y=year_counts.values, 
                     labels={'x': 'Year', 'y': 'Count'}, title="Records by Year")
        st.plotly_chart(fig)
    
    with col2:
        fig = px.bar(x=month_counts.index, y=month_counts.values, 
                     labels={'x': 'Month', 'y': 'Count'}, title="Records by Month")
        st.plotly_chart(fig)

# Daily PM10 patterns page
elif page == "Daily PM10 Patterns":
    st.header("Daily Pattern of PM10 Concentrations")
    
    # Calculate daily averages for time series
    daily_pm10 = data.groupby(['station', pd.Grouper(key='datetime', freq='D')])['PM10'].mean().reset_index()
    
    # Filter for our stations of interest
    station_daily = daily_pm10[daily_pm10['station'].isin(['Dongsi', 'Wanliu'])]
    
    # Display time series
    st.subheader("Daily Average PM10 (2013-2017)")
    
    fig = px.line(station_daily, x='datetime', y='PM10', color='station',
                  title='Daily Average PM10 at Dongsi and Wanliu Stations (2013-2017)')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='PM10',
        legend_title='Station'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate hourly averages
    hourly_pm10 = data[data['station'].isin(['Dongsi', 'Wanliu'])].groupby(['station', 'hour'])['PM10'].mean().reset_index()
    
    # Hourly pattern plot
    st.subheader("Average PM10 by Hour of Day")
    
    fig = go.Figure()
    dongsi_hourly = hourly_pm10[hourly_pm10['station'] == 'Dongsi']
    wanliu_hourly = hourly_pm10[hourly_pm10['station'] == 'Wanliu']
    
    fig.add_trace(go.Scatter(x=dongsi_hourly['hour'], y=dongsi_hourly['PM10'],
                             mode='lines+markers', name='Dongsi Station',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=wanliu_hourly['hour'], y=wanliu_hourly['PM10'],
                             mode='lines+markers', name='Wanliu Station',
                             line=dict(color='red', width=2)))
    
    fig.update_layout(
        title='Hourly Pattern of PM10 at Dongsi and Wanliu Stations',
        xaxis_title='Hour of Day',
        yaxis_title='Average PM10',
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        legend=dict(y=0.99, x=0.99, xanchor='right', yanchor='top'),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of week analysis
    st.subheader("PM10 by Day of Week")
    
    # Group PM10 by day of week to identify weekly patterns
    dow_pm10 = data[data['station'].isin(['Dongsi', 'Wanliu'])].groupby(['station', 'day_of_week'])['PM10'].mean().reset_index()
    
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    dow_pm10['day_name'] = dow_pm10['day_of_week'].map(day_names)
    
    fig = px.bar(dow_pm10, x='day_name', y='PM10', color='station', barmode='group',
                title='Average PM10 by Day of Week at Dongsi and Wanliu Stations')
    
    fig.update_layout(
        xaxis_title='Day of Week',
        yaxis_title='Average PM10',
        xaxis={'categoryorder': 'array', 'categoryarray': list(day_names.values())}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly pattern analysis
    st.subheader("Monthly PM10 Patterns")
    
    # Group data by station, year, month
    monthly_pm10 = data[data['station'].isin(['Dongsi', 'Wanliu'])].groupby(['station', 'year', 'month'])['PM10'].mean().reset_index()
    monthly_pm10['month_year'] = pd.to_datetime(monthly_pm10[['year', 'month']].assign(day=1))
    
    fig = px.line(monthly_pm10, x='month_year', y='PM10', color='station',
                 title='Monthly Average PM10 at Dongsi and Wanliu Stations (2013-2017)')
    fig.update_layout(
        xaxis_title='Month-Year',
        yaxis_title='Average PM10'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight about the patterns
    st.info("""
    **Insights about PM10 Daily Patterns:**
    
    1. **Hourly Pattern:**
       - Higher pollution levels are observed in the evening (8-10 PM)
       - Lower pollution levels are seen during the day (12-2 PM)
       - Both stations show similar patterns, suggesting common pollution sources
       
    2. **Day of Week Pattern:**
       - Higher PM10 levels are observed on weekends (Saturday and Sunday)
       - Weekday pollution levels are relatively consistent
       - Similar patterns at both stations indicate regional pollution factors
       
    3. **Monthly/Seasonal Pattern:**
       - Both stations show almost identical patterns, suggesting similar environmental factors
       - No clear long-term trend from 2013 to 2017
       - Seasonal variations are visible with higher pollution in winter months
    """)

elif page == "Meteorological Correlations":     
    st.header("Correlation Between Meteorological Factors and PM10")
    
    st.write("Is there a correlation between meteorological factors (TEMP, DEWP and PRES) and PM10 concentration levels at the two stations (Dongsi and Wanliu)?")
    
    # Create tabs for the different stations
    tab1, tab2, tab3 = st.tabs(["Dongsi", "Wanliu", "Compare Stations"])
    
    # Define meteorological factors
    meteo_factors = ['TEMP', 'DEWP', 'PRES']
    
    # Filter data for each station
    dongsi_data = data[data['station'] == 'Dongsi']
    wanliu_data = data[data['station'] == 'Wanliu']
    
    with tab1:
        st.subheader("Dongsi Station Correlations")
        
        # Create three columns for the three meteorological factors
        cols = st.columns(3)
        
        # Plot scatter plots for each factor at Dongsi
        for i, factor in enumerate(meteo_factors):
            # Calculate correlation coefficient
            corr = dongsi_data['PM10'].corr(dongsi_data[factor])
            
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4, 4))
                # Create scatter plot with regression line
                sns.regplot(x=factor, y='PM10', data=dongsi_data, ax=ax, 
                           scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
                plt.title(f'{factor} vs PM10\nCorrelation: {corr:.3f}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.write(f"**Correlation with {factor}:** {corr:.3f}")
                
        # Heatmap for Dongsi
        st.subheader("Correlation Heatmap - Dongsi")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dongsi_data[['PM10', 'TEMP', 'DEWP', 'PRES']].corr(),
                   annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Wanliu Station Correlations")
        
        # Create three columns for the three meteorological factors
        cols = st.columns(3)
        
        # Plot scatter plots for each factor at Wanliu
        for i, factor in enumerate(meteo_factors):
            # Calculate correlation coefficient
            corr = wanliu_data['PM10'].corr(wanliu_data[factor])
            
            with cols[i]:
                fig, ax = plt.subplots(figsize=(4, 4))
                # Create scatter plot with regression line
                sns.regplot(x=factor, y='PM10', data=wanliu_data, ax=ax, 
                           scatter_kws={"alpha": 0.3}, line_kws={"color": "red"})
                plt.title(f'{factor} vs PM10\nCorrelation: {corr:.3f}')
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                st.write(f"**Correlation with {factor}:** {corr:.3f}")
                
        # Heatmap for Wanliu
        st.subheader("Correlation Heatmap - Wanliu")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(wanliu_data[['PM10', 'TEMP', 'DEWP', 'PRES']].corr(),
                   annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    
    with tab3:
        st.subheader("Comparison Between Stations")
        
        # Create a multi-select for which correlations to display
        selected_factors = st.multiselect(
            "Select meteorological factors to compare",
            meteo_factors,
            default=meteo_factors
        )
        
        if selected_factors:
            # Create a dataframe to compare correlations
            corr_data = {
                'Factor': [],
                'Dongsi Correlation': [],
                'Wanliu Correlation': []
            }
            
            for factor in selected_factors:
                dongsi_corr = dongsi_data['PM10'].corr(dongsi_data[factor])
                wanliu_corr = wanliu_data['PM10'].corr(wanliu_data[factor])
                
                corr_data['Factor'].append(factor)
                corr_data['Dongsi Correlation'].append(dongsi_corr)
                corr_data['Wanliu Correlation'].append(wanliu_corr)
            
            corr_df = pd.DataFrame(corr_data)
            
            # Display as a table
            st.dataframe(corr_df.style.format({
                'Dongsi Correlation': '{:.3f}',
                'Wanliu Correlation': '{:.3f}'
            }))
            
            # Plot as a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x = np.arange(len(selected_factors))
            width = 0.35
            
            rects1 = ax.bar(x - width/2, corr_df['Dongsi Correlation'], width, label='Dongsi')
            rects2 = ax.bar(x + width/2, corr_df['Wanliu Correlation'], width, label='Wanliu')
            
            ax.set_ylabel('Correlation with PM10')
            ax.set_title('Correlation Comparison Between Stations')
            ax.set_xticks(x)
            ax.set_xticklabels(selected_factors)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add correlation values on top of bars
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            autolabel(rects1)
            autolabel(rects2)
            
            st.pyplot(fig)
        
        # Compare PM10 levels between stations using boxplot
        st.subheader("PM10 Distribution Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='station', y='PM10', data=data[data['station'].isin(['Dongsi', 'Wanliu'])], ax=ax)
        plt.title('Comparison of PM10 Levels Between Dongsi and Wanliu Stations')
        plt.ylabel('PM10')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    # Key insights section
    st.header("Key Insights")
    
    st.write("""
    ### Correlation Analysis:
    
    **Dongsi Station:**
    - TEMP (Temperature) vs PM10: Weak negative correlation (-0.134)
    - DEWP (Dew Point) vs PM10: Weak positive correlation (0.060)
    - PRES (Pressure) vs PM10: Very weak negative correlation (-0.014)
    
    **Wanliu Station:**
    - TEMP (Temperature) vs PM10: Weak negative correlation (-0.119)
    - DEWP (Dew Point) vs PM10: Weak positive correlation (0.055)
    - PRES (Pressure) vs PM10: Very weak negative correlation (-0.021)
    
    ### Pattern Observations:
    - Correlation patterns at both stations are almost identical, suggesting similar atmospheric conditions
    - Temperature shows a consistent negative correlation with PM10 at both stations
    - Dew point shows a consistent positive correlation with PM10 at both stations
    - Pressure shows minimal negative correlation with PM10 at both stations
    - The correlation between temperature and air pressure was consistently negative, while temperature and dew point showed a strong positive relationship at both stations
    
    ### Conclusion:
    The meteorological factors examined (temperature, dew point, and pressure) show similar correlation patterns with PM10 at both stations, but these correlations are relatively weak. This suggests that while these factors may have some influence on PM10 concentrations, other factors not captured in this analysis likely play more significant roles in determining PM10 levels in these areas.
    """)

elif page == "Further Analysis":
    st.header("Further Analysis of Air Quality Data")
    
    # Compare PM10 levels between stations using boxplot
    st.subheader("Comparison of PM10 Levels Between Dongsi and Wanliu Stations")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='station', y='PM10', data=data[data['station'].isin(['Dongsi', 'Wanliu'])], ax=ax)
    ax.set_title('Comparison of PM10 Levels Between Dongsi and Wanliu Stations')
    ax.set_ylabel('PM10')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
    ### Insights from Station Comparison:
    - **Similar Patterns**: There is no significant difference in PM10 levels between Dongsi and Wanliu stations
    - **High Fluctuations**: Both stations experienced high fluctuations with some extreme pollution events
    - **Common Sources**: The source of air pollution at both locations may be from the same factors, such as environmental conditions or human activities
    """)
    
    # Add additional analysis options
    st.subheader("Additional Analysis Options")
    
    analysis_option = st.selectbox(
        "Choose an analysis to view:",
        ["PM10 Distribution", "Yearly Trends", "Statistical Summary"]
    )
    
    if analysis_option == "PM10 Distribution":
        st.subheader("PM10 Distribution at Both Stations")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(data=data[data['station'] == 'Dongsi'], x='PM10', kde=True, label='Dongsi', alpha=0.6, ax=ax)
        sns.histplot(data=data[data['station'] == 'Wanliu'], x='PM10', kde=True, label='Wanliu', alpha=0.6, ax=ax)
        ax.set_title('Distribution of PM10 at Dongsi and Wanliu Stations')
        ax.set_xlabel('PM10')
        ax.set_ylabel('Frequency')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("""
        ### PM10 Distribution Insights:
        - Both stations show similar distribution patterns
        - The majority of PM10 readings are concentrated in the lower range
        - There are numerous outliers indicating episodes of high pollution
        """)
        
        # Calculate yearly averages
        yearly_pm10 = data[data['station'].isin(['Dongsi', 'Wanliu'])].groupby(['station', 'year'])['PM10'].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='year', y='PM10', hue='station', data=yearly_pm10, marker='o', markersize=10, ax=ax)
        ax.set_title('Yearly Average PM10 at Dongsi and Wanliu Stations (2013-2017)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Average PM10')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("""
        ### Yearly Trends Insights:
        - Both stations show similar year-to-year patterns
        - No clear long-term increasing or decreasing trend is observed
        - Some years show higher pollution levels than others, which may be related to policy changes or environmental factors
        """)

        # Create statistical summary
        dongsi_stats = data[data['station'] == 'Dongsi']['PM10'].describe()
        wanliu_stats = data[data['station'] == 'Wanliu']['PM10'].describe()
        
        # Create a dataframe for comparison
        stats_df = pd.DataFrame({
            'Dongsi': dongsi_stats,
            'Wanliu': wanliu_stats
        })
        
        st.dataframe(stats_df)
        
        st.markdown("""
        ### Statistical Summary Insights:
        - Both stations have similar statistical distributions
        - The maximum values indicate extreme pollution events
        - The median (50%) values are significantly lower than the mean, suggesting a right-skewed distribution with occasional high pollution spikes
        """)

elif page == "Summary":
    st.header("Summary of Air Quality Analysis")
    
    st.markdown("""
    ### Key Findings:
    
    #### 1. Daily Pattern of PM10 Concentrations:
    - **Hourly Pattern**: Both stations show similar hourly patterns with:
      - Highest pollution levels in the evening (8-10 PM)
      - Lowest pollution levels during the day (12-2 PM)
    - **Daily Pattern**: PM10 levels fluctuated sharply over the period 2013-2017
    - **Weekly Pattern**: Air quality tends to be worse on weekends (Days 5-6)
    - **Monthly Pattern**: Both stations show almost identical monthly patterns, suggesting similar pollution sources
    
    #### 2. Correlation Between Meteorological Factors and PM10:
    
    **Dongsi Station:**
    - Temperature (TEMP): Weak negative correlation (-0.134)
    - Dew Point (DEWP): Weak positive correlation (0.060)
    - Pressure (PRES): Weak negative correlation (-0.014)
    
    **Wanliu Station:**
    - Temperature (TEMP): Weak negative correlation (-0.119)
    - Dew Point (DEWP): Weak positive correlation (0.055)
    - Pressure (PRES): Weak negative correlation (-0.021)
    
    **Pattern:** Both stations show almost identical correlations, indicating that atmospheric conditions at both sites have similar characteristics.
    
    #### 3. Additional Insights:
    - There is no significant difference in PM10 levels between Dongsi and Wanliu stations
    - Both stations experienced high fluctuations with some extreme pollution events
    - Both stations show similar distributions of PM10 values
    - No clear long-term increasing or decreasing trend in PM10 levels from 2013 to 2017
    
    ### Conclusions:
    - The air quality patterns at both Dongsi and Wanliu stations are remarkably similar, suggesting they are affected by the same pollution sources
    - Meteorological factors have weak correlations with PM10 levels, indicating that other factors (such as human activities) may have more significant impacts on air pollution
    - The daily and weekly patterns suggest that human activities (such as traffic and industrial operations) may be major contributors to air pollution
    - The lack of a clear trend from 2013-2017 suggests that air pollution remains an ongoing issue in these areas
    """)
    
    st.subheader("Recommendations for Further Research:")
    st.markdown("""
    1. Investigate the specific sources of pollution that contribute to the evening peak in PM10 levels
    2. Analyze the impact of traffic patterns on air quality
    3. Study the relationship between industrial activities and PM10 concentrations
    4. Examine the effectiveness of any air quality improvement policies implemented during the study period
    5. Expand the analysis to include other pollutants (PM2.5, NO2, SO2) for a more comprehensive understanding of air quality
    """)
