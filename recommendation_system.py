import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load Dataset
file_path = "https://raw.githubusercontent.com/Anjali2003-dot/Recommendation-System/main/Ecommerce_Sales_Prediction_Dataset.csv"
df = pd.read_csv(file_path)
df
# Display Basic Info
print("\n🔹 Initial Data Overview:")
print(df.info())
print("\n🔹 Descriptive Statistics:")
print(df.describe(include='all'))
df
# Handling Missing Values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Numeric columns
df.fillna(df.mode().iloc[0], inplace=True)  # Categorical columns
df.isnull().sum()
# Step 1: Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # coerce to handle invalid formats

# Step 2: Encode only non-Date object columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'Date':  # 👈 exclude Date
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Remove Duplicates
df.drop_duplicates(inplace=True)
# Convert 'Date' column to datetime format
if 'Date' in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
# Feature Scaling (Standardization)
scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
# Outlier Detection & Removal (Using IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
df
# Remove Infinite and NaN Values
df = df.copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df
# Statistical Analysis
print("\n🔹 Correlation Matrix:")
print(df.corr())
# Select only numeric columns before type conversion
df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(include=[np.number]).astype(np.float32)
df.info()
# Final Shape
print("\n✅ Final Data Shape:", df.shape)
# Set Style
sns.set(style="whitegrid")
# Histogram for Data Distribution
plt.figure(figsize=(12, 6))
df.hist(figsize=(12, 8), bins=30, edgecolor="black")
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()
# Boxplot to Check Outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, palette="coolwarm")
plt.title("Boxplot of Features (Outlier Detection)", fontsize=16)
plt.xticks(rotation=45)
plt.show()
# Pairplot for Feature Relationships (Only Numeric Columns)
df_numeric = df.select_dtypes(include=[np.number])
sns.pairplot(df_numeric, diag_kind="hist", corner=True)  # hist instead of kde to avoid errors
plt.suptitle("Pairplot of Features", fontsize=16)
plt.show()
# Heatmap for Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation
correlation_matrix = df.corr()
# Visualize correlation using heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Units_Sold'])  # Input Features
y = df['Units_Sold']  # Target Variable (Sales)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df.columns)
import pandas as pd

# Convert to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Use .apply() to get UNIX timestamp in seconds
df['Date'] = df['Date'].apply(lambda x: x.timestamp())
# Convert datetime column to UNIX timestamp in seconds
X_train['Date'] = X_train['Date'].apply(lambda x: x.timestamp())
X_test['Date'] = X_test['Date'].apply(lambda x: x.timestamp())

print(X_train.dtypes)
# model.fit(X_train, y_train)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print(df.dtypes)
df = df.drop(columns=['Date'])
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Units_Sold'])  # Features
y = df['Units_Sold']  # Target Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
import pandas as pd

# Group by 'Customer_Segment' to find most sold products
recommendations = df.groupby('Customer_Segment')['Product_Category'].agg(lambda x: x.value_counts().index[0])

# Function to recommend a product based on customer segment
def recommend_product(segment):
    return recommendations.get(segment, "No Recommendation Available")

# Example: Recommend product for a customer segment (e.g., segment '2')
segment = 1
print(f"Recommended Product for Segment {segment}: {recommend_product(segment)}")
df['Customer_Segment'].value_counts()
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate

# Define Reader
reader = Reader(rating_scale=(1, 5))

# Load dataset
data = Dataset.load_from_df(df[['Customer_Segment', 'Product_Category', 'Units_Sold']], reader=reader)

# Apply SVD Model
model = SVD()
cross_validate(model, data, cv=5)
# Train the model on full data
trainset = data.build_full_trainset()
model.fit(trainset)

# Predict for a new user (Example: Customer Segment 2, Product Category 3)
pred = model.predict(uid=2, iid=3)
print(pred.est)  # Estimated rating (higher = better recommendation)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
recommended_products = df.groupby('Product_Category')['Units_Sold'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(x=recommended_products.index, y=recommended_products.values, hue=recommended_products.index, dodge=False,  palette="viridis")
plt.xlabel("Product Category")
plt.ylabel("Total Units Sold")
plt.title("Top Recommended Products")
plt.xticks(rotation=45)
plt.show()
from surprise.model_selection import cross_validate

cv_results = cross_validate(model, data, cv=5)
print(cv_results)
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader

# ✅ Title
st.title("🛒 AI-Powered Recommendation System")

# ✅ Load Data only once
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\anjal\Downloads\Ecommerce_Sales_Prediction_Dataset.csv")
    return df

df = load_data()  # 🟢 load once using cache

# ✅ Show optional preview
if st.checkbox("Show Dataset Preview"):
    st.dataframe(df.head())

# ✅ Build recommendation model
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['Customer_Segment', 'Product_Category', 'Units_Sold']], reader=reader)
model = SVD()
trainset = data.build_full_trainset()
model.fit(trainset)

# ✅ User input
segment = st.number_input("Enter Customer Segment:", min_value=0, max_value=5, step=1)

# ✅ Recommend on click
if st.button("Recommend"):
    if segment in df['Customer_Segment'].unique():
        recommended_product = df[df['Customer_Segment'] == segment]['Product_Category'].mode().values
        if recommended_product.size > 0:
            st.success(f"Recommended Product for Segment {segment}: {recommended_product[0]}")
        else:
            st.warning("No Recommendation Available")
    else:
        st.error("Segment not found in data.")
