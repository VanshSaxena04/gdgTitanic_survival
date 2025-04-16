# gdgTitanic_survival
A ML made with titanic dataset to predict survival rate of victims 

1. pandas, numpy, matplotlib.pyplot, and seaborn are imported for handling data, making calculations, and creating visualizations. StandardScaler, LogisticRegression, accuracy_score, and classification_report from sklearn are imported for preprocessing the data, training the model, and evaluating its performance.

2. The train.csv and test.csv files are loaded into pandas DataFrames called train_df and test_df. This allows us to work with the Titanic data easily. train_df.shape and test_df.shape show the dimensions of both the training and test datasets (number of rows and columns).

3. The Age column is checked for missing values (NaN). Any missing values in the 'Age' column are replaced with the median age of the passengers. The Embarked column is checked for missing values. If any are found, they are replaced with the most common embarkation point using .mode()[0]. The Cabin column is dropped because it has many missing values and may not be useful for analysis. The column names are renamed to make them more descriptive and easier to understand (e.g., 'Survived' becomes 'Survival_Status').

4. A subset of female passengers who survived is created by filtering df for rows where the Sex is 'female' and the Survival_Status is 1 (indicating survival). The gender_survival line groups the data by Sex and calculates the average survival rate for each gender (the mean of Survival_Status). The top_fare line sorts the data by Fare in descending order and selects the top 5 passengers who paid the most. The class_gender_group groups the data by both Passenger_Class and Sex, then calculates the survival rate for each combination of class and gender.

5. The survived_np line converts the Survival_Status column into a Numpy array, which is useful for performing numerical operations. total_np is created as an array of ones, which has the same shape as survived_np. This is used to calculate the survival ratio. The survival_ratio line calculates the proportion of survivors by dividing the sum of survivors by the total number of passengers. age_array holds the 'Age' column values as a Numpy array. The mean and standard deviation of the age_array are printed to give insights into the age distribution.

6. The first plot uses Seaborn's countplot to show how many passengers survived (1) or did not survive (0), broken down by gender (male/female). The second plot uses Seaborn’s histplot to show the distribution of ages for survivors and non-survivors. It includes a kernel density estimate (KDE) to show the age distribution more clearly. The third plot uses Seaborn's barplot to show the survival rate for each passenger class.

7. The categorical variables Sex and Embarked are transformed into numerical features using pd.get_dummies(), which creates new binary columns (1 or 0) for each category. The drop_first=True argument ensures we don't create redundant columns. The test dataset (test_df) is adjusted to match the columns of the training dataset (train_df). Missing columns are filled with 0s to ensure consistency between the training and testing data. The features (X_train and X_test) are separated from the target variable (y_train), which is the survival status. Any missing values in X_test are filled using the median value of the training data, ensuring that the model doesn't receive NaN values.

8. scaling of data has been made beacuse of some error that kept popping in and the online solution was the only way around it
