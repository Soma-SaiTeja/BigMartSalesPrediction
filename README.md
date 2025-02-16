# BigMartSalesPrediction
Best Predicted Test Values

Big Mart Sales – Random Forest Approach: One-Page Summary
1. Goal & Initial Analysis
•	Objective: Predict Item_Outlet_Sales (target) for items across multiple outlets.
•	Data:
o	Train (~8.5k rows) with Item_Outlet_Sales.
o	Test (~5.7k rows) missing Item_Outlet_Sales.
Observations:
1.	Missing Values in Item_Weight and Outlet_Size.
2.	Zero Item_Visibility likely means missing rather than actual zero.
3.	Inconsistent Labels in Item_Fat_Content (LF, low fat, reg).
4.	Skewed target with outliers, leading us to consider a log transform.
________________________________________
2. Problems Faced & Code Updates
1.	Encoding Issue:
o	Problem: After label-encoding Item_Identifier and Outlet_Identifier, we ended up with numeric IDs in the final submission, causing “unknown ID” errors on the leaderboard.
o	Update: We introduced a separate test_original to keep the original IDs intact for the final CSV. Meanwhile, we used an encoded test copy for the model predictions.
2.	Aggregator Merges for Test:
o	Problem: When merging Item_Mean_Sales or Outlet_Mean_Sales from training, some item/outlet combos in test didn’t exist in train, leading to NaN.
o	Update: Filled aggregator NaNs with the global mean of Item_Outlet_Sales, ensuring no missing merges remain.
3.	Zero Visibility:
o	Problem: “0” in Item_Visibility was nonsensical, hurting model performance.
o	Update: Replaced zeros with median visibility grouped by Item_Identifier.
4.	Skewed Sales:
o	Problem: A few extremely large Item_Outlet_Sales values overshadowed the distribution.
o	Update: Used Log_Sales = log1p(Item_Outlet_Sales) to reduce outlier impact and generally lower RMSE.
________________________________________


3. Feature Engineering Steps
1.	Outlet Age
o	train['Outlet_Age'] = 2023 - train['Outlet_Establishment_Year']
o	Rationale: Older outlets often differ in footfall or revenue pattern.
2.	Aggregator Features
o	Item_Mean_Sales: mean sales of each item across outlets.
o	Outlet_Mean_Sales: mean sales of each outlet across items.
o	Hypothesis: some items are inherently popular, some outlets are known to have higher average sales.
o	Merged into train and test; missing combos in test replaced by global mean.
3.	Standard Scaling
o	For numeric columns (Item_Weight, Item_Visibility, Item_MRP, aggregator columns) to help the random forest or advanced boosting methods treat numeric features uniformly.
________________________________________
4. Random Forest Model
1.	Why Random Forest:
o	An ensemble of decision trees, can capture non-linear relationships. Less prone to overfitting than a single tree.
2.	Implementation:
o	Label-encoded categorical columns (except we retain test_original for final IDs).
o	Trained RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42).
o	Used a hold-out validation approach (80/20) to measure local RMSE.
o	Typically ended around 1000–1100 RMSE initially; aggregator features + log transform improved it further (~900–1000 range).
3.	Final Submission:
o	Predictions made on the encoded test.
o	CSV file used IDs from test_original to avoid the “unknown ID” error.
________________________________________
5. Key Takeaways
1.	Separate “test_original” vs. “test” for label encoding to preserve original IDs in submission.
2.	Advanced features (Outlet_Age, aggregator means, log transform) significantly boost performance.
3.	Random Forest provided a solid baseline among ensemble methods, easily improved by further hyperparameter tuning or switching to advanced boosters (CatBoost/XGBoost).
Summary:
We faced encoding & aggregator-merge issues, overcame them by retaining test_original for submission and using aggregator placeholders for missing merges. After final improvements, our Random Forest approach reached a respectable RMSE on local validation and gave us a strong foundation for Big Mart Sales predictions.
I gave a try to other models such as CatBoost and XGBoost to try out if they can achieve best rmse score and run code similarly for them, But Random Forest stood out comparing with them.
CatBoost:
•	Native categorical handling => no heavy manual encoding.
•	Usually the best performer once aggregator features + log transform were applied, especially for data with many categorical variables.
XGBoost:
•	Captured more complex relationships than Random Forest with boosting approach.
•	Required parameter tuning (learning rate, depth, etc.) but generally outperformed RF with advanced features.

