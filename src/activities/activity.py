# Importing libraries.
import time
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from models.ml_models_datasets import MLModels

def init_sidebar_content():
    '''
    Description: 
            Method for initialization the sidebar content (mainly navigation options).
            
    Parameters: 
            None
    
    Returns: 
           navigation - A string containing the navigation option. 
    '''
    st.sidebar.title("Navigate")
    navigation = st.sidebar.radio("", ["Home", "Activity Mode", "About"])
    return navigation

def eda(df):
    '''
    Description: 
            Method that provides various EDA options.
            
    Parameters: 
            df - A pandas dataframe.
    
    Returns: 
           Nothing. 
    '''
    rows, columns = df.shape[0], df.shape[1]
    st.info(f'Rows = {rows}, Columns = {columns}')
    if st.checkbox('Show Target Classes and Value Counts'):
        target_classes = df.target.value_counts()
        st.dataframe(target_classes)
    if st.checkbox("Show DataFrame"):
        num_rows = st.number_input(label="Enter number of rows", min_value=5, max_value=rows)
        st.dataframe(df.head(num_rows))
    if st.checkbox("Describe The Data"):
        st.dataframe(df.describe())
    if st.checkbox("Show DataFrame By Specific Columns"):
        column_names = st.multiselect("Select Columns", df.columns)
        st.dataframe(df[column_names])
    if st.checkbox("Show Data Types"):
        st.dataframe(df.dtypes)

def vis(df):
    '''
    Description: 
            Method for various visualization options.
            
    Parameters: 
            df - A pandas dataframe.
    
    Returns: 
           Nothing. 
    '''
    if st.checkbox("Correlational Matrix"):
        with st.spinner('Generating A Correlational Matrix...'):
            time.sleep(3)
        sns.heatmap(df.corr(), annot=True)
        st.pyplot()
    if st.checkbox("Value Counts"):
        with st.spinner('Generating A Value Count Plot...'):
            time.sleep(3)
        df.target.value_counts().plot(kind='barh')
        st.pyplot()
    if st.checkbox("Pair Plot"):
        with st.spinner('Generating A Pair Plot...'):
            time.sleep(3)
        sns.pairplot(df, hue='target')
        st.pyplot()
    if st.checkbox("Pie Chart"):
        with st.spinner('Generating A Pie Chart...'):
            time.sleep(3)
        df.target.value_counts().plot.pie(autopct='%1.2f%%')
        st.pyplot()
    if st.checkbox('Scatter Plot'):
        x_val = st.selectbox('Select a column for x-axis', df.columns)
        y_val = st.selectbox('Select a column for y-axis', df.columns)
        with st.spinner('Generating A Scatter Plot...'):
            time.sleep(3) 
        plt.scatter(df[x_val], df[y_val], c=df.target)
        plt.xlabel(x_val)
        plt.ylabel(y_val)
        st.pyplot()

def ml(df):
    '''
    Description: 
            Method for handling all the machine learning options.
            
    Parameters: 
            df - A pandas dataframe.
    
    Returns: 
           Nothing. 
    '''
    def run_ml_model(model_name):
        '''
        Description: 
                An inner method for running a machine learning model.
            
        Parameters: 
                model_name - A machine learning model name as a string.
    
        Returns: 
                Nothing.
        '''
        if model_name == 'Linear Regression':
            st.image('res//images//LinearRegression.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            lin_reg = clf.linear_regression()
            lin_reg.fit(x_train, y_train)
            coeff = lin_reg.coef_
            intercept = lin_reg.intercept_
            st.success(f'The coefficients = {coeff}')
            st.success(f'The intercept = {intercept}')
            st.write('Now make an equation of the form y = a1*x1 + a2*x2 + ... an*xn + c')
            st.write('and plugin the features and compare the value you get with the actual target value.')
            st.info('NOTE: Linear Regression is not for classification problems. Hence, use it for Boston Houses or Diabetes dataset to understand this algorithm deeply.')
        elif model_name == 'Logistic Regression':
            st.image('res//images//LogisticRegression.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            C = st.slider(label='Choose C', min_value=0.1, max_value=5.0)
            log_reg = clf.logistic_regression(C)
            train_and_display_metrics(log_reg)
            if st.button('KFold Cross Validation'):
                run_kfold(log_reg) 
        elif model_name == 'K Nearest Neighbors':
            st.image('res//images//KNN.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            n_neighbors = st.number_input(label='n_neighbors', min_value=5, max_value=100)
            knn = clf.k_nearest_neighbors(n_neighbors)
            train_and_display_metrics(knn)
            if st.button('KFold Cross Validation'):
                run_kfold(knn)
            st.info('NOTE: It is often a good practice to scale the features when using KNN because it uses Eucledian distances. However, this topic comes under feature engineering (intermediate level).')
        elif model_name == 'Naive Bayes (Gaussian)':
            st.image('res//images//NaiveBayes.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            nbg = clf.naive_bayes()
            train_and_display_metrics(nbg)
            if st.button('KFold Cross Validation'):
                run_kfold(nbg)
        elif model_name == 'SVM':
            st.image('res//images//SVM1.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            st.image('res//images//SVM2.png', caption='Mathematical Concept (Kernel). Source: Google Images', width=600, format='PNG')
            C = st.slider(label='Choose C', min_value=0.1, max_value=5.0)
            kernel = st.selectbox('Kernel', ['rbf', 'poly', 'linear'])
            svm = clf.svm(C, kernel)
            train_and_display_metrics(svm)
            if st.button('KFold Cross Validation'):
                run_kfold(svm) 
        elif model_name == 'Decision Tree':
            st.image('res//images//DecisionTree.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            max_depth = st.number_input(label='max_depth', min_value=10, max_value=100)
            dt = clf.decision_tree(max_depth)
            train_and_display_metrics(dt)
            if st.button('KFold Cross Validation'):
                run_kfold(dt) 
        elif model_name == 'Random Forest':
            st.image('res//images//RandomForest.png', caption='Mathematical Concept. Source: Google Images', width=600, format='PNG')
            n_estimators = st.number_input('n_estimators', min_value=100, max_value=1000)
            max_depth = st.number_input(label='max_depth', min_value=10, max_value=100)
            rf = clf.random_forest(n_estimators, max_depth)
            train_and_display_metrics(rf)
            if st.button('KFold Cross Validation'):
                run_kfold(rf) 

    def train_and_display_metrics(model):
        '''
        Description: 
                Method to train the model and display its accuracy.
            
        Parameters: 
                model - A ML model (from sklearn).
    
        Returns: 
                Nothing.
        '''
        model.fit(x_train, y_train)
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        st.success(f'Train accuracy = {accuracy_score(y_train, y_pred_train)*100:.5f}%')
        st.success(f'Test accuracy = {accuracy_score(y_test, y_pred_test)*100:.5f}%')
        if st.button('Show Confusion Matrix'):
            cf_matrix = confusion_matrix(y_test, y_pred_test)
            sns.heatmap(cf_matrix, annot=True)
            st.pyplot()

    def run_kfold(model):
        '''
        Description: 
                Method for running kfold cross validation.
            
        Parameters: 
                model - A ML model (from sklearn).
    
        Returns: 
                Nothing.
        '''
        cv = st.number_input(label='Choose number of folds', min_value=5, max_value=20)
        cv_score = cross_val_score(model,x,y, cv=cv)
        sum = 0
        for s in cv_score:
            sum += s
        
        avg_score = sum/cv 
        st.write(f'According to {cv} kfolds, the following test accuracies have been recorded:')
        st.dataframe(cv_score)
        st.success(f'Average test accuracy = {avg_score*100:.5f}%')

    clf = MLModels()
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model_name = st.selectbox("Choose a model/algorithm", ["Linear Regression", "Logistic Regression", "K Nearest Neighbors", "Naive Bayes (Gaussian)", "SVM", "Decision Tree", "Random Forest"])
    run_ml_model(model_name)
   
def pandas_profiling(df):
    '''
    Description: 
            Method for generating pandas profile.
            
    Parameters: 
            df - A pandas dataframe.
    
    Returns: 
            Nothing.
    '''
    if st.button('Generate Report'):
        profile = ProfileReport(df)
        st_profile_report(profile)
