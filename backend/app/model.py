## for dataset genertion ...we use scikit make_moons
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

def train_decision_tree(params):
    X,y=make_moons(
        n_samples=1000,
        noise=0.25,
        random_state=params.random_state
    )

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=params.random_state,stratify=y
    )

    #create model
    clf=DecisionTreeClassifier(
        criterion=params.criterion,
        splitter=params.splitter,
        max_depth=params.max_depth,
        min_samples_split=params.min_samples_split,
        min_samples_leaf=params.min_samples_leaf,
        min_weight_fraction_leaf=params.min_weight_fraction_leaf,
        max_features=params.max_features,
        max_leaf_nodes=params.max_leaf_nodes,
        min_impurity_decrease=params.min_impurity_decrease,
        class_weight=params.class_weight,
        ccp_alpha=params.ccp_alpha,
        random_state=params.random_state
    )

    #Train
    clf.fit(X_train,y_train)

    #predict
    y_pred=clf.predict(X_test)

    #metrics
    metrics={
        "accuracy": accuracy_score(y_test,y_pred),
        "precision": precision_score(y_test,y_pred),
        "recall": recall_score(y_test,y_pred),
        "f1": f1_score(y_test,y_pred),
        "confusion_matrix": confusion_matrix(y_test,y_pred).tolist()
    }

    return clf, metrics, X_train, y_train