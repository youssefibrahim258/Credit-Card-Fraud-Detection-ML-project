from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline

def train_model(model,param_grid,scaler,sampler,x_train,y_train,sampler_option,ratio):
    """
    Train a model using GridSearchCV within an imbalanced-learn pipeline.

    Args:
        model: ML model to train.
        param_grid: Hyperparameter grid for GridSearchCV.
        scaler: Preprocessing scaler.
        sampler: Over/under sampling method.
        x_train: Training features.
        y_train: Training labels.
        sampler_option: Sampler choice (1=Over, 2=Under, 3=Both).
        ratio: Sampling ratio.

    Returns:
        Fitted GridSearchCV object.
    """
    pipeline = ImbPipeline([
        ("scaler", scaler),
        ("sampler", sampler),
        ("model", model)
    ])

    if sampler_option in [1, 2, 3]:
        param_grid["sampler__sampling_strategy"] = [ratio]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(x_train, y_train)
    return grid
