import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek 

def apply_log_transform(data):
    """
    Apply log1p transformation to numeric features.

    Parameters:
    data (pd.DataFrame): Input dataframe containing 'Amount' and 'Time'.

    Returns:
    pd.DataFrame: Transformed dataframe.
    """
    data = data.copy()

    for col in ['Amount', 'Time']:
        data[col] = np.log1p(data[col])

    return data


def get_sampler(option: int, ratio: float):
    """
    option:
        1 -> RandomOverSampler
        2 -> RandomUnderSampler
        3 -> Under + Over (SMOTETomek)
    ratio:
        desired minority ratio (e.g. 0.02)
    """
    samplers = {
        1: RandomOverSampler(sampling_strategy=ratio, random_state=42),
        2: RandomUnderSampler(sampling_strategy=ratio, random_state=42),
        3: SMOTETomek(sampling_strategy=ratio, random_state=42)  # combines under+over
    }
    return samplers[option]



def choose_processor(option:str):
    """
    Returns a scaler object based on option.
    1 -> MinMaxScaler
    2 -> StandardScaler
    """
    if option=="MinMaxScaler":
        processors=MinMaxScaler()
    else:
        processors=StandardScaler()
        

    return processors
