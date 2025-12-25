from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier

def get_model_grids():
    # Return a dictionary of ML models and their hyperparameter grids for training.
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42
            ),
            "params": {
                "model__C": [0.01, 0.1, 1],
                "model__class_weight": [{0: 0.1, 1: 1}, {0: 0.2, 1: 1}]
            }
        },

        "RandomForest": {
            "model": RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            "params": {
                "model__n_estimators": [50, 100, 200],
                "model__max_depth": [10, 20, None],
                "model__class_weight": [{0: 0.1, 1: 1}, {0: 0.2, 1: 10}]
            }
        },

        "MLP": {
            "model": MLPClassifier(random_state=42),
            "params": {
                "model__hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [0.0001, 0.001, 0.01],
                "model__max_iter": [300, 500],
                "model__solver": ["adam"]
            }
        },

        "VotingClassifier": {
            "model": VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                    ("rf", RandomForestClassifier(random_state=42))
                ],
                voting="soft"
            ),
            "params": {
                "model__weights": [[1, 1], [2, 1]]
            }
        }
    }
