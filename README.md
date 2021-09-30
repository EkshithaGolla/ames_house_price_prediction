# ames-housing
Price prediction on the Ames housing [dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


# Table of Contents
1. **Data Perparation:** `00_DataPrep.ipynb`
    1. Preprocessor class

2. **Exploratory Data Analysis:** `01_EDA.ipynb` 

3. **Feature Engineering:** `02_FeatureEngineering.ipynb`
    1. Feature generation
    2. Feature transformation
    3. Feature encoding
    4. EngineerFeatures class
    5. Target transformation

4. **Model Training:** `03_ModelTraining.ipynb`
    1. Preliminary model selection using Pycaret
    2. ModelContainer class
    3. Baseline model
    4. Hyperparameter tuning
    5. Ensembling

5. **Final Results**

---

## 1. Data Preparation

First we explore the data and come up with a strategy to deal with missing values. 

We note the following details about the features:
1. **We have 'regular' categorical features where a null value represents a missing value**
    * These will be imputed with the _mode_ of the feature

2. **We have 'special' categorical features where a null value carries a meaning as described in the `data_description.txt` file in the dataset**
    * These will be imputed with the string `'None'` which will represent the meaning carried by the null value

3. **We have 'regular' numeric features where a null value represents a missing value**
    * These will be imputed with the _median_ of the feature

4. **We have 'special' numeric features which are tied to other features. For example, if a house does not have a basement, the basement surface area will be 0**
    *   The null values for these features will be imputed with `0` 
5. ** Following new columns are created**
	* **Bathrooms** - FullBath and HalfBath columns can be combined to create a new column Bathrooms 
					- This column will contain the total number of bathroom above grade
	* **BsmtBathrooms** - BsmtFullBath and BsmtHalfBath columns can be combined to create BsmtBathrooms 
						- This column will contain the total number of bathrooms below basement

6. **Finally, we note that `'MSSubClass', 'OverallQual', 'OverallCond'` are actually a categorical feature. This will be converted to a `str` type and treated as a categorical variable during feature engineering**


---

## 2. Exploratory Data Analysis

* **Categorical features such as the bathrooms, basement exposure, quality and condition, rooms above grade are affecting the Sales price:**
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/bathrooms.png">
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/basementCondition.png">
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/basementQuality.png">
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/roomsAboveGrade.png">

* ** Categorical feature Heating have 6 different attributes, and almost 97% of the houses have Heating type GasA.
	Dropping Heating column while feature engineering will help in reducing dimentionality as the data available is of only 1 type and this will not help in predicting the Sale Price. **
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/heating.png>
	
	
* **Numeric features such as garage area, and living room surface area correlate strongly with the sale price:**
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/garageArea.png>
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/livingArea.png>

* **More houses are sold in the summer, peaking in June and July**
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/YrSold.png">

* **An important observation is that several features are skewed. This will be taken into account during feature engineering:**
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/skew1.png">
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/skew2.png">


---

## 3. Feature Engineering

There are many categorical variables in this dataset. We first check if both the test and train sets have the same values in all the categorical variables. This is necessary to ensure the train and test sets get encoded properly. In a real world setting, we need to do our best to ensure that the train set is completely representative of the test set.

One example is the feature `'MSSubClass'` where the train set and test set do not have the same number of values. This would mean that encoding these would give different features. 

The best way to avoid this is go back to our preprocessing step and create a combined dataset, and later separate it again into the original train and test sets after feature engineering. This will ensure that both the train and test sets will have the same features after encoding.

Once the data sets are combined, next step is to drop the columns that are not highly correlated with the Sales Price. This is done by finding the correlation between columns and target variable. Based on the EDA, we can drop some columns like Heating, FullBath, HalfBath etc.


### 3.1 Feature transformations

As noted in the EDA, some features are highly skewed. We write a function called `find_skew()` which shows the features and how skewed they are. A snippet of this dataframe is shown below:

<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/skewed_cols.PNG">

`find_skew()` can also return the names of skewed features. We use this information to apply a boxcox transform to the features using `boxcox1p()`.


### 3.2 Feature encoding

We use `pd.get_dummies()` to automatically select and encode the categorical features. 


### 3.3 Target transformation

It is important is to check whether the target variable is normally distributed or not. If it's not then we need to perform some transformations so as to get it closer to normal distribution.

* Let's first get the target variable and plot it

* Then we can use the fit feature of a distplot to see how closely the target variable fits a normal distribution

* Then we can do the same with a log transformed target variable and look at the results

We can see in the plots below that the transformed `'SalePrice'` is much closer to being normally distributed: 
<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/targetTransformation.png">

---

## 4. Model Training

### 4.1 Ensembling

Stacking machine learning models can be a powerful technique. Here we use a simple stacking method and use a weighted average to get the final predictions. The idea is that we are using different types of algorithms that approach the problem in different ways. Combining our models this way allows us to improve the model. However, we need to consider the real world use cases for this. Ensembling increases the model complexity and thus the explainability. Ensemble models will also have a longer runtime, which may make them less valuable for real world applications. 

The trade-off between a slight improvement in model performance and an increase in complexity should be carefully considered before deploying the model into production. 

---
## Final Results

The final ensemble model gave rank of 1169/4847, a **top 25% score**! 

<img  src="https://github.com/EkshithaGolla/ames_house_price_prediction/blob/main/Images/kaggleResult.PNG" width=800>
