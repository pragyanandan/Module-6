import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def get_site_data(url):
    import urllib.request
    import ssl
    import io
    # Create a SSL context to ignore certificate verification
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as f:
        csv_data = f.read().decode('utf-8')  # Read and decode the response in one step
    return io.StringIO(csv_data)


# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'

data_csv = get_site_data(url)

titanic_data = pd.read_csv(data_csv)

# Select features for clustering
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
titanic_data = titanic_data[features]

# Handle missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
titanic_data_scaled = scaler.fit_transform(titanic_data)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
titanic_data['Cluster'] = kmeans.fit_predict(titanic_data_scaled)

# Analyze the cluster centers
print("Cluster Centers:\n", kmeans.cluster_centers_)

# Add cluster labels to the original data
titanic_data['Cluster'] = kmeans.labels_

# Analyze the distribution of clusters
print(titanic_data.groupby('Cluster').mean())

# Visualize the clusters using a pairplot
sns.pairplot(titanic_data, hue='Cluster', palette='viridis')
plt.show()

'''
Pair Plot Observations:
Cluster 0 could represent middle-class passengers with average age and fare.
Cluster 1 might represent first-class passengers who are older and pay higher fares.
Cluster 2 could represent third-class passengers who are younger and pay lower fares.
Insights:
Socioeconomic Differences: Clusters might reveal differences in socioeconomic status among passengers, with distinct groups for first, second, 
and third-class passengers.
Family Size: Differences in family size and travel companions might emerge, such as a cluster with larger families (higher SibSp and Parch values).
Survival Correlation: Although not directly analyzed here, you could later explore how these clusters correlate with survival rates, 
potentially identifying groups with higher or lower chances of survival.

'''