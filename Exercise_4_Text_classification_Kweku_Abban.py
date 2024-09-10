import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay as cmd, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://raw.githubusercontent.com/zhenliangma/Applied-AI-in-Transportation/master/Exercise_4_Text_classification/Pakistani%20Traffic%20sentiment%20Analysis.csv'
df = pd.read_csv(url)

# Dropping duplicates
df.drop_duplicates(inplace=True)

# Split the data into training and testing sets
x = df['Text']
y = df['Sentiment']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define vectorizers
vectorizers = {
    'CountVectorizer': CountVectorizer(ngram_range=(1, 2), stop_words='english', min_df=20),
    'TfidfVectorizer': TfidfVectorizer(min_df=20, norm='l2', smooth_idf=True, use_idf=True, ngram_range=(1, 1),
                                       stop_words='english'),
    'HashingVectorizer': HashingVectorizer(ngram_range=(1, 2), n_features=200)
}

# Define models and their hyperparameters
models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=0), {'C': [0.001, 0.01, 0.1, 1, 10, 100]}),
    'KNeighborsClassifier': (KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}),
    'RandomForestClassifier': (RandomForestClassifier(random_state=0),
                               {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30],
                                'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}),
    'XGBClassifier': (
    XGBClassifier(), {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300], 'max_depth': [3, 4, 5]}),
    'SVC': (SVC(probability=True), {'kernel': ['linear', 'rbf', 'poly'], 'C': [0.1, 1, 10]}),
    'BernoulliNB': (BernoulliNB(), {'alpha': [0.1, 0.5, 1], 'force_alpha': [True, False]})
}

# Store results
results = []

# Iterate over vectorizers
for vec_name, vectorizer in vectorizers.items():
    # Vectorize data
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    # Iterate over models
    for model_name, (model, param_grid) in models.items():
        # Perform Grid Search with 5-fold cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(x_train_vectorized, y_train)

        # Extract the best parameters and scores
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        test_accuracy = grid_search.best_estimator_.score(x_test_vectorized, y_test)

        # Store results
        results.append({
            'Vectorizer': vec_name,
            'Model': model_name,
            'Best Params': best_params,
            'CV Best Score': best_score,
            'Test Accuracy': test_accuracy
        })

        # Display confusion matrix
        cmd.from_estimator(grid_search.best_estimator_, x_test_vectorized, y_test,
                           display_labels=['Positive', 'Negative'], cmap='Blues', xticks_rotation='vertical')
        plt.show()

# Display all results in a DataFrame for easy comparison
results_df = pd.DataFrame(results)
print(results_df)

# Identify the best model configuration
best_model_config = results_df.loc[results_df['Test Accuracy'].idxmax()]
print("\nBest Model Configuration:\n", best_model_config)
