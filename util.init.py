#Change the width of a jupyter notebook
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:80% !important; }</style>"))

#Standard Libraries
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames
import random
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
#User counter to check frequencies (https://pymotw.com/2/collections/counter.html)
from  collections import Counter
import re

#Visualisation Libraries
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from pandas.plotting import scatter_matrix
import seaborn as sns
# Pretty display for notebooks
%matplotlib inline
# Import supplementary visualizations code visuals.py
#import visuals as vs

#Modelling Libraries
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import tree
from sklearn.metrics import fbeta_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler