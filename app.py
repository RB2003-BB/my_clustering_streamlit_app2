# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 16:18:19 2025

@author: LAB
"""

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#load model
with open('kmeans_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
#set the page of config
st.set_page_config(page_title = "K-Means Clustering", layout= "centered")   
 
#set the title
st.title("K-Means Clustering Visualizer by Rapeepan Srisuwan")



#load dataset
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#predict 
y_kmeans = loaded_model.predict(X)

#plot
fig, ax = plt.subplots()
plt = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(loaded_model.cluster_centers_[:, 0],loaded_model.cluster_centers_[:, 1], s=300, c='red')
ax.set_title('K-Mean Clustering')
ax.legend()
st.pyplot(fig)

