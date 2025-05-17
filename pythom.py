import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Load dataset (replace with your Kaggle dataset path)
df = pd.read_csv('data.csv', encoding='ISO-8859-1')

# Data Preprocessing
df.dropna(inplace=True)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Task 1: Seasonal Trend Analysis
plt.figure(figsize=(12, 6))
df.groupby('YearMonth')['TotalPrice'].sum().plot(marker='o', linestyle='-')
plt.title('Total Sales Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Revenue')
plt.xticks(rotation=45)
plt.show()

# Task 2: Customer Lifetime Value (CLV) Features
customer_value = df.groupby('CustomerID').agg({'TotalPrice': 'sum', 'InvoiceDate': ['min', 'max']})
customer_value.columns = ['TotalSpend', 'FirstPurchase', 'LastPurchase']
customer_value['CustomerLifetime'] = (customer_value['LastPurchase'] - customer_value['FirstPurchase']).dt.days
print(customer_value.head())

# Task 3: Feature Importance for Predicting CLV
# Use Random Forest Regressor to predict CLV (TotalSpend)
X = customer_value[['CustomerLifetime']]  # You can include more features if available
y = customer_value['TotalSpend']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42, n_jobs=1)  # Disable parallelism to avoid warnings
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
print(f"Feature importance for predicting CLV: {importances}")

# Task 4: Clustering for Customer Segmentation
features = customer_value[['TotalSpend', 'CustomerLifetime']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA to improve customer segmentation
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

kmeans = KMeans(n_clusters=3, random_state=42)
customer_value['Cluster'] = kmeans.fit_predict(features_pca)
sns.scatterplot(data=customer_value, x=features_pca[:, 0], y=features_pca[:, 1], hue='Cluster', palette='viridis')
plt.title('Customer Segmentation (PCA)')
plt.show()

# Task 5: Association Rule Mining (Apriori)
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)  # Binary transformation for apriori
freq_items = apriori(basket, min_support=0.01, use_colnames=True)  # Lower min_support
rules = association_rules(freq_items, metric='lift', min_threshold=1)
print(rules.head())

# Task 6: Comparing Recommendation Algorithms

# Collaborative Filtering using KNN
# Nearest Neighbors (based on item similarity)
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
knn.fit(basket.T)  # Item-based CF

# Example: Find nearest neighbors for an item (replace with actual item index)
distances, indices = knn.kneighbors(basket.T.iloc[0].values.reshape(1, -1))

# Content-Based Filtering (using cosine similarity on item features)
item_features = basket.T  # You could use actual features if available
cos_sim = cosine_similarity(item_features)
print(cos_sim)

# Task 7: Promotional Event Impact
event_period = (df['InvoiceDate'] >= '2022-11-25') & (df['InvoiceDate'] <= '2022-11-30')
promo_sales = df[event_period].groupby('YearMonth')['TotalPrice'].sum()
norm_sales = df[~event_period].groupby('YearMonth')['TotalPrice'].sum()

plt.plot(promo_sales.index.astype(str), promo_sales.values, label='Promo Period', marker='o')
plt.plot(norm_sales.index.astype(str), norm_sales.values, label='Normal Period', marker='s')
plt.legend()
plt.title('Impact of Promotions on Sales')
plt.show()

# Task 8: Visualizing Customer Journey Paths (Sankey Diagram)
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["Browsing", "Add to Cart", "Purchase", "Payment"]
    ),
    link=dict(
        source=[0, 1, 2],
        target=[1, 2, 3],
        value=[8, 4, 2]
    ))])

fig.update_layout(title_text="Customer Journey Flow", font_size=10)
fig.show()

print("Analysis Completed Successfully!")
