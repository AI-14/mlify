# Importing libraries.
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits, load_boston, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Defining ML models class.
class MLModels:
    def linear_regression(self):
        '''
        Description: 
                Method for creating a linear regression classifier.
             
        Parameters: 
                None
    
        Returns: 
                clf - A linear regression classifier.  
        '''
        clf = LinearRegression()
        return clf

    def logistic_regression(self, C):
        '''
        Description: 
                Method for creating a logistic regression classifier.
            
        Parameters: 
                C - Inverse of regularization strength.
    
        Returns: 
                clf - A logistic regression classifier.
        '''
        clf = LogisticRegression(C=C)
        return clf

    def k_nearest_neighbors(self, n_neighbors):
        '''
        Description: 
                Method for creating a knn classifier.
            
        Parameters: 
                n_neighbors - Number of neighbors to use.
    
        Returns: 
                clf - A knn classifier.
        '''
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        return clf

    def naive_bayes(self):
        '''
        Description: 
                Method for creating a naive-bayes classifier.
            
        Parameters: 
                None
    
        Returns: 
                clf - A naive-bayes classifier.
        '''
        clf = GaussianNB()
        return clf

    def svm(self, C, kernel):
        '''
        Description: 
                Method for creating a svm classifier.
            
        Parameters: 
                C - Regularization parameter.
                kernel - Specifies the kernel type to be used in the algorithm.
    
        Returns: 
                clf - A svm classifier.
        '''
        clf = SVC(C=C, kernel=kernel)
        return clf

    def decision_tree(self, max_depth):
        '''
        Description: 
                Method for creating a decision tree classifier.
            
        Parameters: 
                max_depth - The maximum depth of the tree.
    
        Returns: 
                clf - A decision tree classifier.
        '''
        clf = DecisionTreeClassifier(max_depth=max_depth)
        return clf

    def random_forest(self, n_estimators, max_depth):
        '''
        Description: 
                Method for creating a decision tree classifier.
            
        Parameters: 
                n_estimatos - The number of trees in the forest.
                max_depth - The maximum depth of the tree.
    
        Returns: 
                clf - A random forest classifier.
        '''
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        return clf

    
# Defining dataset class.
class MLDataset:
    def __init__(self, dataset_name=None):
        '''
        Description: 
                Method that initializes the dataset name.
            
        Parameters: 
                dataset_name - The name of the dataset.
    
        Returns: 
                Nothing
        '''
        self.dataset_name = dataset_name
    
    def get_dataframe(self):
        '''
        Description: 
                Method to get a pandas dataframe based on the dataset name initialized in __init__().
            
        Parameters: 
                None
    
        Returns: 
                df - A pandas dataframe.
        '''
        df = None
        if self.dataset_name == "Iris":
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
        elif self.dataset_name == "Breast Cancer":
            bc = load_breast_cancer()
            df = pd.DataFrame(bc.data, columns=bc.feature_names)
            df['target'] = bc.target    
        elif self.dataset_name == 'Wine Quality':
            wq = load_wine()
            df = pd.DataFrame(wq.data, columns=wq.feature_names)
            df['target'] = wq.target
        elif self.dataset_name == 'Mnist Digits':
            dig = load_digits()
            df = pd.DataFrame(dig.data, columns=dig.feature_names)
            df['target'] = dig.target
        elif self.dataset_name == 'Boston Houses':
            bh = load_boston()
            df = pd.DataFrame(bh.data, columns=bh.feature_names)
            df['target'] = bh.target
        elif self.dataset_name == 'Diabetes':
            db = load_diabetes()
            df = pd.DataFrame(db.data, columns=db.feature_names)
            df['target'] = db.target
        return df   