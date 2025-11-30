import pandas as pd
from sklearn.datasets import load_iris

def load_iris_df():
    raw = load_iris()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)
    df['target'] = raw.target
    df['target_name'] = [raw.target_names[i] for i in raw.target]
    return df
