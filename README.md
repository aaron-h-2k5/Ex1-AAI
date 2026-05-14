<H3> Name: Aaron H </H3>
<H3>Register No. 212223040001</H3>
<H3> Experiment 1</H3>
<H3>DATE: 14.05.26</H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>
## Aim :
    To create a bayesian Network for the given dataset in Python
## Algorithm:
Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```
import pandas as pd # for data manipulation
import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
# for creating Bayesian Belief Networks (BBN)
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
#Set Pandas options to display more columns
pd.options.display.max_columns=50

# Read in the weather data csv
# Load dataset
df = pd.read_csv('weatherAUS.csv', encoding='utf-8')

# Drop rows where target is missing
df = df[df['RainTomorrow'].notnull()]

# Fill missing values with mean (numeric columns)
df = df.fillna(df.mean(numeric_only=True))

# Create categorical bands
df['WindGustSpeedCat'] = df['WindGustSpeed'].apply(
    lambda x: '<=40' if x <= 40 else '40-50' if x <= 50 else '>50'
)

df['Humidity9amCat'] = df['Humidity9am'].apply(
    lambda x: '>60' if x > 60 else '<=60'
)

df['Humidity3pmCat'] = df['Humidity3pm'].apply(
    lambda x: '>60' if x > 60 else '<=60'
)
def probs(data, child, parent1=None, parent2=None):
    if parent1 is None:
        prob = pd.crosstab(data[child], 'count', normalize='columns') \
                .sort_index().to_numpy().reshape(-1).tolist()

    elif parent2 is None:
        prob = pd.crosstab(data[parent1], data[child], normalize='index') \
                .sort_index().to_numpy().reshape(-1).tolist()

    else:
        prob = pd.crosstab([data[parent1], data[parent2]], data[child],
                           normalize='index') \
                .sort_index().to_numpy().reshape(-1).tolist()

    return prob

     
H9am = BbnNode(
    Variable(0, 'H9am', ['<=60', '>60']),
    probs(df, child='Humidity9amCat')
)

H3pm = BbnNode(
    Variable(1, 'H3pm', ['<=60', '>60']),
    probs(df, child='Humidity3pmCat', parent1='Humidity9amCat')
)

W = BbnNode(
    Variable(2, 'W', ['<=40', '40-50', '>50']),
    probs(df, child='WindGustSpeedCat')
)

RT = BbnNode(
    Variable(3, 'RT', ['No', 'Yes']),
    probs(df, child='RainTomorrow',
          parent1='Humidity3pmCat',
          parent2='WindGustSpeedCat')

bbn = Bbn() \
    .add_node(H9am) \
    .add_node(H3pm) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9am, H3pm, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3pm, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

# Convert to Join Tree
join_tree = InferenceController.apply(bbn)

pos = {
    0: (-1, 2),
    1: (-1, 0.5),
    2: (1, 0.5),
    3: (0, -1)
}

# Graph styling
options = {
    "font_size": 16,
    "node_size": 4000,
    "node_color": "white",
    "edgecolors": "black",
    "edge_color": "red",
    "linewidths": 2,
}

# Draw graph
n, labels = bbn.to_nx_graph()
nx.draw(n, with_labels=True, labels=labels, pos=pos, **options)

plt.axis("off")
plt.show()



```

## Output:
<img width="691" height="495" alt="Screenshot 2026-05-14 at 2 38 28 PM" src="https://github.com/user-attachments/assets/6c13a58a-f6fc-4ee8-baa4-edff43a86303" />


## Result:
   Thus a Bayesian Network is generated using Python

