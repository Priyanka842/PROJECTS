import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
penguins_df = pd.read_csv("penguins.csv")

print(penguins_df.head())

# Remove missing values
penguins_df = penguins_df.dropna()

# Convert sex column
penguins_df = pd.get_dummies(
    penguins_df,
    columns=["sex"],
    drop_first=True
)

# Scale data
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_df)

# Elbow method
inertias = []

for k in range(1, 10):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(penguins_scaled)

    inertias.append(model.inertia_)

# Plot elbow curve
plt.plot(range(1, 10), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_method.png")
plt.show()

# KMeans
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

penguins_df["label"] = kmeans.fit_predict(penguins_scaled)

# Statistics dataframe
stat_penguins = (
    penguins_df.groupby("label")[
        [
            "culmen_length_mm",
            "culmen_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]
    ]
    .mean()
)

print("\nCluster Statistics:\n")
print(stat_penguins)