Predicting Airbnb Prices with NYC Crime Data
Tools: Python, Pandas, Scikit-learn, BallTree

In this project, I analyzed and integrated two real-world datasets, 2019 NYC Airbnb listings and NYC crime reports, to explore the impact of local crime density on Airbnb pricing.

Key Steps:
Data Cleaning & Preparation: Cleaned 48,000+ Airbnb records and ~200,000 crime incidents. Removed irrelevant features and filtered extreme values.
Geospatial Data Integration: Used Haversine distance with BallTree to count crimes within a 200-meter radius of each listing.

Exploratory Data Analysis:
Visualized price distributions across boroughs and room types.
Found a weak positive Pearson correlation (0.10) between local crime count and Airbnb price.

Predictive Modeling:
Built and optimized K-Nearest Neighbors Regressor using GridSearchCV (MAE: 61.5, R²: 0.22).
Tuned a Random Forest Regressor for comparison (MAE: 61.7, R²: 0.21).
Included encoded categorical variables, standard scaling, and hyperparameter tuning.

Key Takeaways:
Crime density alone isn’t a strong predictor of price: While the correlation between local crime count and Airbnb prices was weak (Pearson r ≈ 0.10), it highlighted the importance of combining multiple factors such as room type, availability, and reviews for more accurate price predictions.

Modeling reinforces the value of feature engineering: Including both spatially integrated features and encoded categorical variables improved the model performance, showing that feature design can boost accuracy.

Hands-on experience with full ML workflow: Gained practical experience in data cleaning, feature engineering, model selection, hyperparameter tuning, and performance evaluation using KNN and Random Forest regressors on real-world datasets.
