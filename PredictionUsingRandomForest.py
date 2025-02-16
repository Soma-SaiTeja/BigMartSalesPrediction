{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4dd0b046-dc3d-4bac-a4c0-e982d1f1ba0f",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:28: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:29: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test['Item_Weight'].fillna(test['Item_Weight'].median(), inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  train['Outlet_Size'].fillna('Unknown', inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:33: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test['Outlet_Size'].fillna('Unknown', inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:36: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  train['Item_Fat_Content'].replace({'low fat': 'Low Fat','LF': 'Low Fat','reg': 'Regular'}, inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:37: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test['Item_Fat_Content'].replace({'low fat': 'Low Fat','LF': 'Low Fat','reg': 'Regular'}, inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:77: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test['Item_Mean_Sales'].fillna(global_mean_sales, inplace=True)\n",
      "C:\\Users\\somas\\AppData\\Local\\Temp\\ipykernel_33532\\3462552895.py:78: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.\n",
      "The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.\n",
      "\n",
      "For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.\n",
      "\n",
      "\n",
      "  test['Outlet_Mean_Sales'].fillna(global_mean_sales, inplace=True)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Random Forest] RMSE: 1005.2302\n",
      "✅ Submission file saved at: C:\\Users\\somas\\Documents\\BigMart Sales Prediction\\BigMart_Sales_Predictions.csv\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "\"\\n# If you want to see a more robust RMSE across repeated folds:\\nfrom sklearn.model_selection import RepeatedKFold, cross_val_score\\n\\nrkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)\\nscores = cross_val_score(\\n    rf_model, X, y,\\n    scoring='neg_root_mean_squared_error',\\n    cv=rkf,\\n    n_jobs=-1\\n)\\n\\nprint('RepeatedKFold Mean RMSE:', -scores.mean())\\n# This helps confirm if your model is stable across multiple folds\\n\""
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "\n",
    "# (Optional) XGBoost if you'd like advanced boosting\n",
    "# import xgboost as xgb\n",
    "\n",
    "\n",
    "# 1. LOAD DATASETS\n",
    "\n",
    "train = pd.read_csv('C:/Users/somas/Documents/BigMart Sales Prediction/train_v9rqX0R.csv')\n",
    "test_original = pd.read_csv('C:/Users/somas/Documents/BigMart Sales Prediction/test_AbJTz2l.csv')\n",
    "\n",
    "# Copy test for transformations\n",
    "test = test_original.copy()\n",
    "\n",
    "# 2. DATA CLEANING\n",
    "# Missing Item_Weight => median\n",
    "train['Item_Weight'].fillna(train['Item_Weight'].median(), inplace=True)\n",
    "test['Item_Weight'].fillna(test['Item_Weight'].median(), inplace=True)\n",
    "\n",
    "# Missing Outlet_Size => 'Unknown'\n",
    "train['Outlet_Size'].fillna('Unknown', inplace=True)\n",
    "test['Outlet_Size'].fillna('Unknown', inplace=True)\n",
    "\n",
    "# Standardize Item_Fat_Content\n",
    "train['Item_Fat_Content'].replace({'low fat': 'Low Fat','LF': 'Low Fat','reg': 'Regular'}, inplace=True)\n",
    "test['Item_Fat_Content'].replace({'low fat': 'Low Fat','LF': 'Low Fat','reg': 'Regular'}, inplace=True)\n",
    "\n",
    "# Zero Item_Visibility => median by Item_Identifier\n",
    "visibility_median = train.groupby('Item_Identifier')['Item_Visibility'].median()\n",
    "train.loc[train['Item_Visibility'] == 0, 'Item_Visibility'] = \\\n",
    "    train['Item_Identifier'].map(visibility_median)\n",
    "test.loc[test['Item_Visibility'] == 0, 'Item_Visibility'] = \\\n",
    "    test['Item_Identifier'].map(visibility_median)\n",
    "\n",
    "# 3. FEATURE ENGINEERING\n",
    "\n",
    "# Outlet Age\n",
    "train['Outlet_Age'] = 2023 - train['Outlet_Establishment_Year']\n",
    "test['Outlet_Age'] = 2023 - test['Outlet_Establishment_Year']\n",
    "\n",
    "train.drop('Outlet_Establishment_Year', axis=1, inplace=True)\n",
    "test.drop('Outlet_Establishment_Year', axis=1, inplace=True)\n",
    "\n",
    "# AGGREGATE FEATURES (ITEM & OUTLET) \n",
    "# Let's create 2 aggregator features from the training set:\n",
    "#   (1) mean sales per Item_Identifier\n",
    "#   (2) mean sales per Outlet_Identifier\n",
    "\n",
    "item_mean_sales = train.groupby('Item_Identifier')['Item_Outlet_Sales'].mean().reset_index()\n",
    "item_mean_sales.columns = ['Item_Identifier','Item_Mean_Sales']\n",
    "\n",
    "outlet_mean_sales = train.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().reset_index()\n",
    "outlet_mean_sales.columns = ['Outlet_Identifier','Outlet_Mean_Sales']\n",
    "\n",
    "# Merge these aggregates into train\n",
    "train = pd.merge(train, item_mean_sales, on='Item_Identifier', how='left')\n",
    "train = pd.merge(train, outlet_mean_sales, on='Outlet_Identifier', how='left')\n",
    "\n",
    "# We'll do the same merges for test\n",
    "test = pd.merge(test, item_mean_sales, on='Item_Identifier', how='left')\n",
    "test = pd.merge(test, outlet_mean_sales, on='Outlet_Identifier', how='left')\n",
    "\n",
    "# NA might appear in test merges if an ID doesn't exist in train. Fill with global mean.\n",
    "global_mean_sales = train['Item_Outlet_Sales'].mean()\n",
    "test['Item_Mean_Sales'].fillna(global_mean_sales, inplace=True)\n",
    "test['Outlet_Mean_Sales'].fillna(global_mean_sales, inplace=True)\n",
    "\n",
    "\n",
    "# 4. LABEL ENCODING\n",
    "\n",
    "encoder = LabelEncoder()\n",
    "categorical_cols = [\n",
    "    'Item_Identifier', 'Item_Fat_Content', 'Item_Type',\n",
    "    'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'\n",
    "]\n",
    "\n",
    "for col in categorical_cols:\n",
    "    train[col] = encoder.fit_transform(train[col])\n",
    "    test[col] = encoder.transform(test[col])\n",
    "\n",
    "\n",
    "# 5. LOG TRANSFORM THE TARGET\n",
    "\n",
    "# Convert target to log-space to reduce outlier impact\n",
    "train['Log_Sales'] = np.log1p(train['Item_Outlet_Sales'])\n",
    "\n",
    "# 6. DEFINE FEATURES & TARGET\n",
    "\n",
    "# We'll drop 'Item_Outlet_Sales' in favor of 'Log_Sales' for training\n",
    "X = train.drop(columns=['Item_Outlet_Sales','Log_Sales'])\n",
    "y = train['Log_Sales']   # log-transformed target\n",
    "\n",
    "\n",
    "# 7. TRAIN-TEST SPLIT\n",
    "\n",
    "X_train, X_valid, y_train, y_valid = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "\n",
    "# 8. SCALING NUMERIC FEATURES\n",
    "\n",
    "scaler = StandardScaler()\n",
    "num_cols = [\n",
    "    'Item_Weight','Item_Visibility','Item_MRP','Outlet_Age',\n",
    "    'Item_Mean_Sales','Outlet_Mean_Sales'  # newly added aggregator columns\n",
    "]\n",
    "\n",
    "X_train[num_cols] = scaler.fit_transform(X_train[num_cols])\n",
    "X_valid[num_cols] = scaler.transform(X_valid[num_cols])\n",
    "test[num_cols]     = scaler.transform(test[num_cols])\n",
    "\n",
    "\n",
    "# 9. RANDOM FOREST MODEL\n",
    "\n",
    "rf_model = RandomForestRegressor(\n",
    "    n_estimators=200, max_depth=10, random_state=42\n",
    ")\n",
    "\n",
    "rf_model.fit(X_train, y_train)\n",
    "rf_preds_log = rf_model.predict(X_valid)\n",
    "\n",
    "# Evaluate with RMSE in normal space\n",
    "rf_preds_normal = np.expm1(rf_preds_log)  # revert log->normal\n",
    "y_valid_normal  = np.expm1(y_valid)\n",
    "\n",
    "rf_rmse = np.sqrt(mean_squared_error(y_valid_normal, rf_preds_normal))\n",
    "print(f'[Random Forest] RMSE: {rf_rmse:.4f}')\n",
    "\n",
    "\n",
    "# 10. TRAIN ON FULL DATA & PREDICT\n",
    "\n",
    "rf_model.fit(X, y)\n",
    "test_preds_log = rf_model.predict(test)\n",
    "test_preds_normal = np.expm1(test_preds_log)\n",
    "\n",
    "\n",
    "# 11. SUBMISSION\n",
    "\n",
    "# Create submission with original IDs (test_original)\n",
    "submission = test_original[['Item_Identifier', 'Outlet_Identifier']].copy()\n",
    "submission['Item_Outlet_Sales'] = test_preds_normal\n",
    "\n",
    "project_dir = os.getcwd()\n",
    "submission_file_path = os.path.join(project_dir, \"BigMart_Sales_Predictions.csv\")\n",
    "\n",
    "submission.to_csv(submission_file_path, index=False)\n",
    "print(f'✅ Submission file saved at: {submission_file_path}')\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "978d00d4-271c-4e32-94a8-56682556aa1d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
