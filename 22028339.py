import pdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit



def Clustering_of_CO2_emissions_per_capita_2018_2019():
    '''Reading  through pandas'''
    df = pd.read_csv("Climate_change.csv", skiprows=4)
    '''Indicators selection'''
    co2_per_capita = df[df['Indicator Name'] == 'CO2 emissions (metric tons per capita)']
    gdp_per_capita = df[df['Indicator Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)']

    # Merge the CO2 and GDP dataframes on the 'Country Name' column
    merged_df = pd.merge(co2_per_capita, gdp_per_capita, on='Country Name')

    # Extract the relevant columns for clustering
    data = merged_df[['Country Name', '2018_y', '2019_y']].dropna()
    '''Data is large so selecting only 2 countries'''
    # Prepare the data for clustering
    X = data[['2018_y', '2019_y']]
    X = (X - X.mean()) / X.std()  # Normalize the data
    '''Performing K mean '''
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # Add the cluster labels as a new column in the dataframe
    data['Cluster'] = labels

    '''Plotting 1st graph'''
    plt.scatter(data['2018_y'], data['2019_y'], c=data['Cluster'], cmap='viridis')
    plt.xlabel('CO2 emissions per capita (2018)')
    plt.ylabel('CO2 emissions per capita (2019)')
    plt.title('Clustering of CO2 emissions per capita')
    plt.colorbar(label='Cluster')
    plt.show()


    ''' Calculate confidence ranges using the err_ranges function'''
def err_ranges(fit_params, fit_cov, x, confidence=0.95):
    alpha = 1 - confidence
    n = len(x)
    p = len(fit_params)
    t_score = abs(alpha/2)
    err = np.sqrt(np.diag(fit_cov))
    lower = fit_params - t_score * err
    upper = fit_params + t_score * err
    return lower, upper


Clustering_of_CO2_emissions_per_capita_2018_2019()



# Load the climate change indicators dataset
climate_change_df = pd.read_csv("Climate_change.csv", skiprows=4)

# Display the available indicators
print(climate_change_df["Indicator Name"].unique())

'''Select relevant indicators'''
indicators_of_interest = [
    "CO2 emissions (metric tons per capita)",
    "Total greenhouse gas emissions (kt of CO2 equivalent)"
]



'''Filter the climate change dataframe based on the selected indicators'''
filtered_df = climate_change_df[climate_change_df["Indicator Name"].isin(indicators_of_interest)]

# Pivot the dataframe to convert indicators into columns
pivoted_df = filtered_df.pivot(index="Country Name", columns="Indicator Name", values="2019")

# Drop rows with missing values
pivoted_df.dropna(inplace=True)

# Display the pivoted dataframe
print(pivoted_df.head())

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(pivoted_df.values)

'''Convert the normalized data back to a dataframe'''
normalized_df = pd.DataFrame(normalized_data, columns=pivoted_df.columns, index=pivoted_df.index)

# Display the normalized dataframe
print(normalized_df.head())

'''Perform K-means clustering'''
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(normalized_df.values)

# Add the cluster labels to the dataframe
normalized_df["Cluster"] = clusters

# Display the dataframe with cluster labels
print(normalized_df.head())

# Define the two indicators for visualization
x_indicator = "CO2 emissions (metric tons per capita)"
y_indicator = "Total greenhouse gas emissions (kt of CO2 equivalent)"

# Plot the clusters
plt.scatter(
    normalized_df[x_indicator], normalized_df[y_indicator],
    c=normalized_df["Cluster"], cmap="viridis"
)

plt.xlabel(x_indicator)
plt.ylabel(y_indicator)
plt.title("Clustering based on Climate")
plt.show()




# Define the function for curve fitting
def exponential_growth(x, a, b):
    return 1000
    # return a * np.exp(b * x)

# Select a country for curve fitting
# country_data = climate_change_df[climate_change_df['Country Name'] == 'United States']

check = climate_change_df.loc[climate_change_df['Indicator Name']== 'CO2 emissions (metric tons per capita)']
check1 = check.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'], axis = 1)



new_df1 = climate_change_df.loc[climate_change_df['Indicator Name']== 'Total greenhouse gas emissions (kt of CO2 equivalent)']
new_df2 = new_df1.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'], axis = 1)

check1 = check1.fillna('')
new_df2 = new_df2.fillna('')


x = check1.values
x_new = []
for data in x:
    # print('')
    for data1 in data:
        if data1 != '':
            x_new.append(data1)

x_new = x_new[:7000]
y = new_df2.values
y_new = []

for data in y:
    # print('')
    for data1 in data:
        if data1 != '':
            y_new.append(data1)

y_new = y_new[:7000]


# y1 = country_data['2000']
# Fit the data using the exponential growth model


params, pcov = curve_fit(exponential_growth, x_new, y_new)

# Make predictions for future years
future_years = np.arange(2023, 2043)
predicted_values = exponential_growth(future_years, x_new[:20], y_new[:20])
confidence_range = 1.96 * np.sqrt(np.diag(pcov))



# Plot the best fitting function and confidence range
# plt.figure(figsize=(10, 6))
# plt.plot(x_new, y_new, 'bo', label='Actual Data')
# plt.plot(future_years[:1], predicted_values, 'r-', label='Best Fitting Function')
# plt.fill_between(future_years[:1], predicted_values - confidence_range, predicted_values + confidence_range,
#                  color='gray', alpha=0.3, label='Confidence Range')
# plt.xlabel('Year')
# plt.ylabel('CO2 emissions (metric tons per capita)')
# plt.title('Exponential Growth Model for CO2 Emissions')
# plt.legend()
# plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(x_new, y_new, label='Data')
plt.plot(x_new[:1], exponential_growth(x_new, *params), 'r-', label='Best Fit')
plt.plot(x_new, y_new, 'g--', label='Predictions')
plt.fill_between(x_new, y_new, y_new, color='gray', alpha=0.3, label='Confidence Interval')
plt.xlabel('Time')
plt.ylabel('Attribute')
plt.title('Exponential Growth Model')
plt.legend()
plt.grid(True)
plt.show()

scaler = StandardScaler()
df_std = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns, index=normalized_df.index)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(normalized_df)
centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Plot the clusters and centers
plt.scatter(normalized_df.iloc[:, 0], normalized_df.iloc[:, 1], c=clusters)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.xlabel(normalized_df.columns[0])
plt.ylabel(normalized_df.columns[1])
plt.show()

