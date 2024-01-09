
import pandas as pd 
import numpy as np
import seaborn as sns
from scipy.stats import normaltest
from statsmodels.graphics.gofplots import qqplot
import matplotlib.pyplot as plot
import missingno as msno
import plotly.express as px 

class Plotter:
    
    def __init__(self, df_raw):
        self.df_raw = df_raw
        
    def show_null_values(self):
        print(self.df_raw.isnull().sum()/len(self.df_raw), "\n")
        msno.matrix(self.df_raw) 
        
    def test_normal_distribution(self,column_name):    
        data = self.df_raw[column_name]
        stat, p = normaltest(data, nan_policy='omit')
        qq_plot = qqplot(self.df_raw[column_name], scale=1, line='q')
        plot.title(f"The distribution of {column_name} values")
        plot.show()
        print("Statistics=%.3f, p=%.3f" % (stat, p))
        print(f'The median of {column_name} is {self.df_raw[column_name].median()}')
        print(f'The mean of {column_name} is {self.df_raw[column_name].mean()}')
        
    def show_disc_prob_distr(self, column_name):
        plot.rc("axes.spines", top=False, right=False)
        probs = self.df_raw[column_name].value_counts(normalize=True)
        dpd=sns.barplot(y=probs.index, x=probs.values, color='b')
        plot.xlabel("Values")
        plot.ylabel("Probability")
        plot.title(f"Discrete Probability Distribution of {column_name}")
        plot.show()
        print("\nThe value counts in  ")
        print(self.df_raw[column_name].value_counts())
        print(f'\nThe mode of {column_name} column is {self.df_raw[column_name].mode()[0]}')
            
    def show_skewness(self, column_name):
        plot.hist(self.df_raw[column_name], bins=50)
        plot.show()
        print(f"\nSkew of {column_name} column is {self.df_raw[column_name].skew()}", "\n")
            
    def show_cont_data_outliers(self, column_name):
        column_of_interest = column_name
        self.df_raw.boxplot(column=column_of_interest)
        plot.show()
            
    def show_correlation_cont_data(self, columns):
        columns_of_interest = self.df_raw[columns]
        return px.imshow(columns_of_interest.corr(), title="Correlation heatmap")
        
#visuals = Plotter(df_raw)

#visuals.show_null_values()

#visuals.test_normal_distribution()

#visuals.show_disc_prob_distr()

#visuals.show_skewness()

#visuals.show_cont_data_outliers()

#visuals.show_correlation_cont_data()