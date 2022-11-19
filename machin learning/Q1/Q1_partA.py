import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sn

d_train0 = pd.read_csv('./MNIST/mnist_train.csv')
l_train = d_train0['label']  # label 데이터
d_train = d_train0.drop("label", axis=1) # 이미지 데이터

d_test0 = pd.read_csv('./MNIST/mnist_test.csv')
l_test = d_test0['label']  # label 데이터
d_test = d_test0.drop("label", axis=1) # 이미지 데이터

def get_vectors(l,d):
    labels = l
    data = d

    standardized_data = StandardScaler().fit_transform(data) # 데이터 정규화
    sample_data = standardized_data
    covar_matrix = np.matmul(sample_data.T, sample_data)

    values, vectors = eigh(covar_matrix, eigvals=(782,783)) # top2 eigenvalue
    vectors = vectors.T  # (2, 784)
    return labels, vectors


def projection(sample_data, labels, vectors):
    new_coordinates = np.matmul(vectors, sample_data.T)
    new_coordinates = np.vstack((new_coordinates, labels)).T
    print(new_coordinates.shape)
    exit()
    dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
    sn.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.savefig("Q1_Part_A_b.jpg")

if __name__ == '__main__':
    labels, vectors = get_vectors(l_train, d_train)
    labels = l_test
    data = d_test

    standardized_data = StandardScaler().fit_transform(data)  # 데이터 정규화
    sample_data = standardized_data

    projection(sample_data, labels, vectors)
