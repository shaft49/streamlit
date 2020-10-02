import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

st.title('Streamlit Example')
# this is markdown
st.write("""
    # Explore different classifier and datasets
""")

dataset_name = st.sidebar.selectbox('Select Dataset', ['Iris', 'Breast Cancer', 'Wine'])
classifier_name = st.sidebar.selectbox('Select Classifier', ['KNN', 'SVM', 'Random Forest'])

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    
    X = data.data
    y = data.target

    return X, y

X, y = get_dataset(dataset_name)

st.write(f'Shape of dataset: {X.shape}')
st.write(f'No Of Classes: {len(np.unique(y))}')

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'KNN':
        no_of_neighbors = st.sidebar.slider('No of Neighbors', 1, 15)
        weights = st.sidebar.selectbox('Which weight you want to use?', ['uniform', 'distance'])
        params['no_of_neighbors'] = no_of_neighbors
        params['weights'] = weights
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0) # C-Support Vector Classification. Regularization parameter.
        params['C'] = C
    elif clf_name == 'Random Forest':
        max_depth = st.sidebar.slider('Max Depth', 2, 15)
        no_of_estimators = st.sidebar.slider('No Of Estimators', 1, 100)

        params['max_depth'] = max_depth
        params['no_of_estimators'] = no_of_estimators
    return params

params = add_parameter_ui(classifier_name)


def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors = params['no_of_neighbors'], weights = params['weights'])
    elif clf_name == 'SVM':
        clf = SVC(C = params['C'])
    elif clf_name == 'Random Forest':
        clf = RandomForestClassifier(max_depth = params['max_depth'], n_estimators = params['no_of_estimators'], random_state = 1234)
    
    return clf

clf = get_classifier(classifier_name, params)

def plot_data(X, y):
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    X1 = X_projected[:, 0]
    X2 = X_projected[:, 1]
    
    fig = plt.figure()
    plt.scatter(X1, X2, c = y, alpha = 0.8, cmap = 'viridis')
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()

    return fig

plot_data_button = st.button('Plot Dataset')

if plot_data_button:
    fig = plot_data(X, y)
    st.pyplot(fig)

submit_button = st.button('Start Training')



if submit_button:
    ## Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier Name: {classifier_name}')
    st.write(f'Dataset Name: {dataset_name}')
    st.write(f'Accuracy: {acc:0.4f}')