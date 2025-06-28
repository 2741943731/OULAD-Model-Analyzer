# OULAD
[Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset) analysis exercise.

The OULAD dataset contains open source, anonymized data of students, courses, assessments, and virtual learning 
environments (VLE) clickstreams, allowing for analyses into the nexus of student behavior and outcomes. 

## Project outline

#### Research questions
1) What are important predictors for negative course outcomes? (when `final_result` is `withdraw` or `fail`)

#### Project approach and pipeline
1. Initial data cleaning and ETL --> creates initial master dataframes
   1. Read in raw csv files and save as dataframes in `data_dict` dictionary
   2. Initial EDA on raw data - creates summary of each dataframe, dimensions, missing rows, and column names
   3. Creates master dataframes for each of the research questions
2. Additional data cleaning --> Creates cleaned dataframes
   1. Check for missing values
   2. Drop columns where all values are null
   3. Drop records with missing values in non-numeric columns
   4. Impute 0 value into missing values in numeric columns
   5. Check for duplicates rows, drop duplicates if applicable
   6. Check for outliers for numeric variables and cap outliers at the 98th percentile (since data are right skewed)
   7. Prep VLE data by aggregating number of days and clicks for each VLE medium and total so that there are aggregated metrics for each student/course/semester
   8. Feature engineering
      1. Create `year` and `semester` columns from `code_presentation`
      2. Label encode categorical columns 
      3. Create `overall_total_clicks` per student/course/semester
3. EDA and descriptive analyses 
   1. Explore univariate patterns: Create histograms for numeric variables and bar plots for categorical variables
   2. Explore bivariate relationships:
         1. Create correlation matrix for all numeric variables
         2. Create stacked bar plots by `final_result` group for categorical variables
         3. Create grouped box plots by `final_result` group for numeric  variables and group box plots for `score` values by categorical variables
         4. Create scatter plots for `scores` and numeric variables
         5. Create VLE summary info to explore average values and student utilization (percentage) for each VLE medium 
4. Predictive modeling 
   1. Run grid search to determine optimial hyperparameters for model, using 5-fold CV
   2. Save hyperparameters to use for final model
   3. Split training and test sets (80:20)
   4. Train final random forest model (classifier for RQ1 and regressor for RQ2)
   5. Run model predictions on test data set and use results to measure final model performance 
5. Model evaluation
   1. Performance metrics determined and saved to `results` dictionary output
      1. RQ1 - f1 score, AUC score, confusion matrix
      2. RQ2 - MSE, and R<sup>2</sup> 
      3. Feature importances from RF output

Note - Random forests were used because they are robust against multicollinearity and skewness 

## About the data

![alt test](https://analyse.kmi.open.ac.uk/resources/images/model.png)

## Data Summary
Raw data summary 

| Table | Row, Cols | Missing rows | Column names |
| --- | --- | --- | --- | 
| assessments | 206, 6 | 11 | ['code_module', 'code_presentation', 'id_assessment', 'assessment_type', 'date', 'weight'] | 
| courses | 22, 3 | 0 | ['code_module', 'code_presentation', 'module_presentation_length']  | 
| studentAssessment | 173912, 5 | 173 | ['id_assessment', 'id_student', 'date_submitted', 'is_banked', 'score'] | 
| studentInfo | 32593, 12 | 1111 | ['code_module', 'code_presentation', 'id_student', 'gender', 'region', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability', 'final_result'] | 
| studentRegistration | 32593, 5 | 22560 | ['code_module', 'code_presentation', 'id_student', 'date_registration', 'date_unregistration'] | 
| studentVLE | 10655280, 6 | 0 | ['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click'] | 
| vle | 6364, 6 | 5243 | ['id_site', 'code_module', 'code_presentation', 'activity_type', 'week_from', 'week_to'] | 


Cleaned data for RQ1 
- 28,174 records (1 row for each student/course/semester)
- 25,149 unique students, 7 courses from 2 semesters across 2 years (2013 and 2014)
- Outcome variable = `final_result` (i.e., pass, withdraw, fail, distinction)
- 42% Pass, 23% Fail, 10% Distinction, 25% Withdraw
- Data includes aggregated VLE information about student behavior from whole course duration

Notable univariate notes from EDA
- Most VLE variables, number of previous course attempts, were heavily right skewed
- Relatively equal split of gender representation and IMD bands
- Majority of students were between 0 and 35, majority did NOT have a disability
- Majority of students passed the course

Notable bivariate EDA notes
- `final_results` relationships:
  - Students with no formal higher education had the lowest pass and distinction rates, and the highest rates of failing and withdrawing
  - Subtle positive relationship between passing and IMD band, such that the higher the IMD band, the higher the rate of passing and lower the rate of failing 
  - Positive outcomes seem to increase by age band
  - Students with a disability have higher withdraw rates
  - VLE – course performance seems to be positively related to number of days student interacted with the VLE and the average total clicks they used in the VLE
- `score` relationships:
  - Distribution of test scores differed by course,but not by semester/year, gender, region
  - Students with post-grad education had higher assessment performances
  - Subtle positive relationship between score and IMD band, such that the higher the IMD band, the higher the score 
  - Positive relationship between score and age band
  - Students with a disability have slightly lower scores

## Modeling Results

#### RQ1
- Accuracy =  0.83
- F1 score = 0.81
- ROC AUC = 0.83

Based on feature importances and Shapley value outputs, the total number of distinct days that students
interacted with the VLE was the most impactful predictor on `final_result` (followed closely by `n_days_homepage`, `overall_total_clicks`, and `n_days_quiz`).


## Limitations:
 - Did not examine bivariate relationships using statistical tests (e.g., t-test, chi-square, correlations, ANOVA)
 - Did not account for lagged effects of VLE interactions in relation to assessment dates
 - Limited information on certain fields (e.g., imd_band)
 - Did not explore multivariate relationships
 - Did not run comparative tests using other ML approaches for comparison (e.g., XGBoost, OLS regression, baysian regression)
 - Did not explore dimensionality reduction (e.g., PCA or clustering methods) to consolidate variance from VLE fields
 - Poor performance on RQ2 was not evaluated in detailed to improve upon. Could perform a residual analysis to discover what types of feature engineering and tuning could improve performance. 
 - Try one hot encoding (drop_first=True) for `course_module` since not ordinal

## Reference:

[Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics dataset Sci. Data 4:170171 
doi: 10.1038/sdata.2017.171 (2017).](https://pubmed.ncbi.nlm.nih.gov/29182599/)

