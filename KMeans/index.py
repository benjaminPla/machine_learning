from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/four.csv")
df.rename(
    {"Annual Income (k$)": "Income", "Spending Score (1-100)": "Spending Score"},
    axis=1,
    inplace=True,
)

#  sns.pairplot(df, hue="Gender")
#  fig = plt.gcf()
#  plt.savefig('pairplot.pdf', dpi=150)

X = df[["Income", "Spending Score"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(X_scaled)
#  sns.scatterplot(
    #  x=X_scaled[:, 0],
    #  y=X_scaled[:, 1],
    #  hue=kmeans.labels_,
    #  legend="full",
    #  palette=sns.color_palette("hls", 5),
#  )
#  plt.xlabel('Income')
#  plt.ylabel('Spending Score')
#  centroids = kmeans.cluster_centers_
#  plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=50,color='k')
#  plt.savefig('clusters.pdf', dpi=150)

# how to know how many cluster?
# elbow rule
#  sq_distances = []
#  k_values = range(2, 10)
#  for k in k_values:
    #  kmeans = KMeans(n_clusters=k, n_init=10)
    #  kmeans.fit(X_scaled)
    #  sq_distances.append(kmeans.inertia_)
#  sns.lineplot(x=k_values, y=sq_distances, marker="o", size=30, legend=False)
#  plt.xlabel("x")
#  plt.ylabel("y")
#  plt.savefig("elbow.png", dpi=150)

# silhouette
sil = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    sil.append(score)
sns.lineplot(x=k_values, y=sil, marker="o", size=30, legend=False)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("silhouette.png", dpi=150)
