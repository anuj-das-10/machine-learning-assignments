## 1. Introduction

Decision Trees are widely used in machine learning for classification tasks, and ID3 (Iterative Dichotomiser 3) is one of the fundamental algorithms used to construct them. Developed by Ross Quinlan in 1986, the ID3 algorithm builds a decision tree by recursively selecting the best attribute that provides the highest Information Gain based on Entropy.

### 1.1. Key Concepts of ID3
- **Entropy** → A measure of uncertainty or impurity in a dataset.

- **Information Gain** → The reduction in entropy after splitting the dataset based on an attribute.

- **Recursive Splitting** → The algorithm continues selecting attributes and splitting data until it reaches a stopping condition (e.g., pure labels or no attributes left).

<br /><br />

## 2. Flowchart of ID3 Algorithm


![Image](/images/flowchart.png)





























<br /><br />

## 3. Python Source Code

```python

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


```




<br /><br />

## 4. Datasets for Testing
The dataset *“playTennis.csv”*, along with two additional datasets containing approximately **2000 records each**, was used to test the implemented ID3 algorithm.



![Image](/images/playTennis-Dataset.png)




























<br /><br />

## 5. Output for the *“playTennis.csv”* dataset

![Image](/images/playTennis-Output.png)



















<br /><br />

## 6. Reference to other datasets

- Play Tennis
<a href="/datasets/playTennis.csv" download="Play_Tennis.csv">Download Dateset</a>


- Customer Purchase Behaviour
<a href="/datasets/cpb.csv" download="Customer_Purchase_Behaviour.csv">Download Dateset</a>


- Student Performance
<a href="/datasets/sp.csv" download="Student_Performance.csv">Download Dateset</a>
