#Analysing Covid-19 Dataset

import numpy as np
import panda as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Importing Dataset
corona_dataset_csv = pd.read_csv("Datasets/covid19_Confirmed_dataset.csv")
corona_dataset_csv.head(5)
corona_dataset_csv.shape


#Deleting Useless columns
corona_dataset_csv.drop(["Lat", "Long"], axis =1, inplace = True)
corona_dataset_csv.head(5)

#Grouping by the country
corona_dataset_csv.groupby("Countr/Region").sum()

#Visualising data related to country
corona_dataset_csv.loc["India"]

#Compare the data by country with the help of Plotting
corona_dataset_csv.loc["India"].plot()
corona_dataset_csv.loc["China"].plot()
corona_dataset_csv.loc["Italy"].plot()
plt.legend()

#For Plotting the rise by the desired date
corona_dataset_csv.loc["India"][:10].plot()

#Calculating the max. spike of the covid-19 case:
corona_dataset_csv.loc["India"].diff().plot()

#Finding the max. derivative:
countries = list(corona_dataset_csv.index)
max_infection_rates = []
for c in countries:
    max_infection_rates.append(corona_dataset_csv.loc[c].diff().max())
corona_dataset_csv['max infection rate'] = max_infection_rates

#Creating a new Dataframe with needed Dataset
corona_data = pd.DataFrame(corona_dataset_csv['max infection rate'])
corona_data.head()
