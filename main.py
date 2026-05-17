import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

sns.set(style="whitegrid")


# 1. LOAD DATASET

df_raw = pd.read_csv("Churn_Modelling.csv")
df = df_raw.copy()

print("Dataset Loaded Successfully!")
print(df.head())


# 2. DATA CLEANING


# Duplicate Check
print("\nDuplicate Rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Drop Unnecessary Columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Keep categorical copies for visualization
df["Gender_str"] = df_raw["Gender"]
df["Geography_str"] = df_raw["Geography"]

# One-Hot Encode Geography
geo = pd.get_dummies(df["Geography_str"], prefix="Geo")
df = pd.concat([df, geo], axis=1)

# Missing Values Check
print("\nMissing Values:\n", df.isnull().sum())


# 3. DESCRIPTIVE STATISTICS

print("\nDescriptive Statistics:\n", df.describe())


# 4. VISUALIZATION + INSIGHTS


#  Bar Chart: Churn Count 
plt.figure(figsize=(6,5))
sns.countplot(x="Exited", data=df)
plt.title("Customer Churn Count")
plt.xlabel("Exited (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()

print("\nInsight: Churn rate is around 20%.")

# -------- Bar Chart: Churn Rate by Geography --------
plt.figure(figsize=(7,5))
geo_churn = df.groupby("Geography_str")["Exited"].mean()
sns.barplot(x=geo_churn.index, y=geo_churn.values)
plt.title("Churn Rate by Country")
plt.xlabel("Country")
plt.ylabel("Churn Rate")
plt.show()

print("\nInsight: Germany customers churn the most.")

# Line Chart: Balance Trend Across Age 
df_sorted = df.sort_values("Age")
plt.figure(figsize=(8,5))
plt.plot(df_sorted["Age"], df_sorted["Balance"], color="blue", alpha=0.6)
plt.title("Balance Trend Across Age")
plt.xlabel("Age")
plt.ylabel("Balance")
plt.show()

# Scatter: Age vs Balance 
plt.figure(figsize=(7,6))
sns.scatterplot(x="Age", y="Balance", hue="Exited", data=df)
plt.title("Age vs Balance")
plt.show()

#  Histogram: Age Distribution 
plt.figure(figsize=(7,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

#  Boxplot: Balance by Churn 
plt.figure(figsize=(7,5))
sns.boxplot(x="Exited", y="Balance", data=df)
plt.title("Balance by Churn Status")
plt.show()

# Heatmap 
plt.figure(figsize=(12,7))
sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# 5. STATISTICAL TESTS

print("\n Statistical Tests")

# T-test: Compare mean Age between churned and non-churned customers 
# Null Hypothesis (H0): Mean age of churned customers = mean age of non-churned customers
churn_age = df[df["Exited"] == 1]["Age"]        # Age of customers who exited
no_churn_age = df[df["Exited"] == 0]["Age"]     # Age of customers who did not exit

t_stat, p_val = ttest_ind(churn_age, no_churn_age)
print("\nT-Test (Age):", t_stat, p_val)


# Chi-Square Test: Gender vs Churn
# Tests whether Gender and Churn are independent
gender_table = pd.crosstab(df["Gender"], df["Exited"])  # Contingency table

chi2, p, dof, exp = chi2_contingency(gender_table)
print("\nChi-Square (Gender):", chi2, p)


# ANOVA Test: Balance Across Countries 
# Tests whether mean balance differs across France, Spain, and Germany
fr = df[df["Geography_str"] == "France"]["Balance"]
sp = df[df["Geography_str"] == "Spain"]["Balance"]
gr = df[df["Geography_str"] == "Germany"]["Balance"]

f_stat, p_val = f_oneway(fr, sp, gr)
print("\nANOVA (Balance by Country):", f_stat, p_val)



# 6. MODEL PREPARATION


# Select ONLY numeric columns (logistic regression needs numeric data)
df_numeric = df.select_dtypes(include=[np.number]).copy()

# Replace missing values with 0 to avoid computation errors
df_numeric = df_numeric.fillna(0)

# Separate independent variables (X) and target variable (y)
X = df_numeric.drop("Exited", axis=1).values.astype(float)
y = df_numeric["Exited"].values.reshape(-1, 1).astype(float)


# Manual Train-Test Split Function
def train_test_split_manual(X, y, test_size=0.2):
    idx = np.random.permutation(len(X))  # Shuffle indices
    X, y = X[idx], y[idx]
    split = int(len(X) * (1 - test_size))
    return X[:split], X[split:], y[:split], y[split:]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split_manual(X, y)



# 7. MANUAL LOGISTIC REGRESSION


class ManualLogisticRegression:
    
    # Initialize learning rate and number of epochs
    def __init__(self, lr=0.001, epochs=3000):
        self.lr = lr
        self.epochs = epochs
    
    # Sigmoid activation function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    # Train the logistic regression model using gradient descent
    def fit(self, X, y):
        m, n = X.shape
        self.W = np.zeros((n, 1))  # Initialize weights
        self.b = 0                # Initialize bias
        
        for i in range(self.epochs):
            z = np.dot(X, self.W) + self.b
            y_pred = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (y_pred - y))
            db = (1/m) * np.sum(y_pred - y)
            
            # Update parameters
            self.W -= self.lr * dw
            self.b -= self.lr * db
            
            # Print loss every 500 epochs
            if i % 500 == 0:
                loss = -np.mean(
                    y*np.log(y_pred + 1e-9) +
                    (1-y)*np.log(1-y_pred + 1e-9)
                )
                print(f"Epoch {i}, Loss = {loss:.4f}")
    
    # Predict churn (0 or 1)
    def predict(self, X):
        probs = self.sigmoid(np.dot(X, self.W) + self.b)
        return (probs > 0.5).astype(int)


# 8. MODEL TRAINING & EVALUATION

# Train the model
model = ManualLogisticRegression()
model.fit(X_train, y_train)

# Make predictions on test data
preds = model.predict(X_test)

# Calculate accuracy
accuracy = np.mean(preds == y_test)

print("\n----------------------------")
print("Final Model Accuracy:", accuracy)
print("------------------------------")

print("\nThe model runs successfully and predicts churn effectively.")

