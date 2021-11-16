import pandas as pd

df = pd.read_csv("/mnt/local_data/datasets/master-thesis/100000/1634326474/0/real_state.csv")
df.describe().to_csv("describe.csv")
