# Import necessary libraries

import pandas as pd
import lime
import lime.lime_tabular    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imPipeline
import numpy as np

df = pd.read_csv("C:\\Users\\admin\\2nd_assign_mlops\\framingham.csv")

# Overview of the dataset
# dataset link: https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression

print("First 5 rows of the dataset:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nData types and missing values:")
print(df.info())

# Data cleaning
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Handling missing values by dropping rows with missing values 
df_cleaned = df.dropna()
print(f"\nRows remaining after dropping missing values: {df_cleaned.shape[0]}")

# Feature Extraction
# Splitting the features and target variable
X = df_cleaned.drop('TenYearCHD', axis=1)  # Features
y = df_cleaned['TenYearCHD']  # Target

# Standardize the features (scaling/normalization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Printing the shape of the scaled dataset
print(f"Shape of X (features) after scaling: {X_scaled.shape}")
print(f"Shape of y (target): {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Balancing the dataset using SMOTE and RandomUnderSampler
over = SMOTE(sampling_strategy='minority')  # Over-sampling the minority class
under = RandomUnderSampler(sampling_strategy='majority')  # Under-sampling the majority class

# Creating a pipeline
pipeline = imPipeline(steps=[('o', over), ('u', under)])

# Fit and transform the training data
X_train_balanced, y_train_balanced = pipeline.fit_resample(X_train, y_train)

# Printing the shape of the balanced dataset
print(f"\nShape of X_train (features) after balancing: {X_train_balanced.shape}")
print(f"Shape of y_train (target) after balancing: {y_train_balanced.shape}")


profile = ProfileReport(df, title='Pandas Profiling Report', explorative=True)
profile.to_file("EDA_report.html")



import h2o
from h2o.automl import H2OAutoML

# Initializing H2O cluster
h2o.init()

# Convert numpy arrays back to pandas DataFrames for X_train and X_test
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# Concatenate the target column with the feature DataFrame
train_df = pd.concat([X_train_df, y_train.reset_index(drop=True)], axis=1)
test_df = pd.concat([X_test_df, y_test.reset_index(drop=True)], axis=1)

# target column contains 0 and 1
train_df['TenYearCHD'] = train_df['TenYearCHD'].apply(lambda x: 1 if x > 0 else 0)
test_df['TenYearCHD'] = test_df['TenYearCHD'].apply(lambda x: 1 if x > 0 else 0)

# Converting Pandas DataFrame to H2O Frame
h2o_train = h2o.H2OFrame(train_df)
h2o_test = h2o.H2OFrame(test_df)

# Print columns to check the target column
print(h2o_train.columns)

# Specify feature columns and target column
y = 'TenYearCHD'  # Target column
x = h2o_train.columns[:-1]  # Features (all except the target column)

# target is treated as a factor for classification tasks
h2o_train[y] = h2o_train[y].asfactor()

# Initialize and run the H2O AutoML
aml = H2OAutoML(max_models=3, seed=1, verbosity='info')
aml.train(x=x, y=y, training_frame=h2o_train)

# View the AutoML Leaderboard
lb = aml.leaderboard
print("====================================================")
print("Leaderboard of models: ")
print(lb)

# Print the best model (leader)
best_model = aml.leader
print("====================================================")
print("\nBest Model Details:")
print(best_model)

# Evaluate the best model on the test set
preds = best_model.predict(h2o_test[x])
preds_df = preds.as_data_frame()

# Print the first few rows of predictions
print("====================================================")
print("\nPredictions DataFrame:")
print(preds_df.head())

# Extract and print only relevant probabilities (for 'CHD' and 'No CHD')
# Assuming 'p1' is 'CHD' and 'p0' is 'No CHD'
print("====================================================")
print("\nFiltered Prediction Probabilities:")
preds_df_filtered = preds_df[['predict', 'p0', 'p1']]
print(preds_df_filtered.head())

# Print the performance metrics of the best model
perf = best_model.model_performance(h2o_test)
print("====================================================")
print("\nBest Model Performance on Test Data:")
print(perf)

# Print a summary of why it's the best model
best_algorithm = best_model.algo
best_auc = perf.auc()

print("====================================================")
print(f"\nThe best model is a {best_algorithm} model.")
print(f"The model was selected based on its performance, achieving an AUC of {best_auc:.4f} on the test data.")


# Convert H2O Frame to numpy array for LIME
X_train_np = np.array(X_train_df)
X_test_np = np.array(X_test_df)

# Initialize LIME for the best model
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_np,
    feature_names=X_train_df.columns,
    class_names=['No CHD', 'CHD'],
    verbose=True,
    mode='classification'
)

# Choose an instance for LIME explanation 
instance_idx = 12
instance = X_test_np[instance_idx]

# Define a function to get model predictions for LIME
def predict_fn(data):
    # Convert the numpy array back to a Pandas DataFrame
    data_df = pd.DataFrame(data, columns=X_train_df.columns)
    # Convert the DataFrame to an H2O Frame
    data_h2o = h2o.H2OFrame(data_df)
    # Predict using the best model
    predictions = best_model.predict(data_h2o)
    # Extract the probabilities (the column names might be 'p0', 'p1', or similar)
    probs = predictions.as_data_frame().filter(like='p').values
    # Print the prediction DataFrame
    print(predictions.as_data_frame())

    # Inspect specific instance probabilities
    instance_probabilities = preds.as_data_frame().iloc[instance_idx]
    print("Instance probabilities: ", instance_probabilities)

    # Normalize probabilities if they do not sum to 1
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

# Explain the prediction using LIME
lime_exp = lime_explainer.explain_instance(instance, predict_fn)
lime_exp.save_to_file('lime_output.html')
