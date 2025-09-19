
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login

api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/subhash33/tourism-package/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

tourism_dataset['gender'] = tourism_dataset['gender'].replace('Fe Male', 'Female')
# tourism_dataset['NumberOfChildrenVisiting'] = tourism_dataset['NumberOfChildrenVisiting'].astype(int)
# tourism_dataset['Age'] = tourism_dataset['Age'].astype(int)
float_features = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome']
tourism_dataset[float_features] = tourism_dataset[float_features].astype(int)

# Define the target variable for the classification task
target = ['ProdTaken']

# List of numerical features in the dataset
numerical_features = [
    'Age',
    'CityTier',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'Passport',
    'PitchSatisfactionScore',
    'OwnCar',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation',
]

# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numerical_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split('/')[-1],
        repo_id="Subhash33/tourism-package",
        repo_type="dataset",
    )
