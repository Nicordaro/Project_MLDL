import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def update_confusion_matrix(matrix, preds, datas):
    for pred, data in zip(preds,datas):
        matrix[data.item(),pred.item()] = matrix[data.item(),pred.item()]+1
        
def new_confusion_matrix(lenx=100, leny=100):
    matrix = np.zeros((leny, lenx))
    return matrix

def show_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(15,9))
    ax = sns.heatmap(matrix, linewidth=0.2,cmap='Oranges')
    plt.show()
