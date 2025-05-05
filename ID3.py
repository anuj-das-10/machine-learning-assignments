import math
import pandas as pd
from pprint import pprint

# Function to calculate entropy
def entropy(data):
    labels = data.iloc[:, -1].value_counts()
    total = len(data)
    return sum([-count/total * math.log2(count/total) for count in labels])

# Function to find the best attribute to split
def best_attribute(data):
    attributes = data.columns[:-1]  
    gains = {}
    total_entropy = entropy(data)
    
    for attr in attributes:
        values = data[attr].unique()
        weighted_entropy = sum(
            (len(subset) / len(data)) * entropy(subset)
            for value in values
            for subset in [data[data[attr] == value]]
        )
        gains[attr] = total_entropy - weighted_entropy

    return max(gains, key=gains.get)

# Recursive ID3 function
def id3(data):
    if len(data.iloc[:, -1].unique()) == 1:
        return data.iloc[0, -1]
    
    if len(data.columns) == 1:
        return data.iloc[:, -1].mode()[0]

    best_attr = best_attribute(data)
    tree = {best_attr: {}}
    
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value].drop(columns=[best_attr])
        tree[best_attr][value] = id3(subset)
    
    return tree

# Load dataset from .txt or .csv file
filename = input("Enter file path/name: ")
try:
    data = pd.read_csv(filename)
    decision_tree = id3(data)
    print("Decision Tree: ", end="")
    pprint(decision_tree)
except Exception as e:
    print("Error:", e)
