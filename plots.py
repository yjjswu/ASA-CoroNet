import math
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt

from evaluation import compute_performance_measures

#%%

df_cm = pd.read_csv('fig/cm.csv', index_col=0)

#%%

df_cm

#%%

df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
fig = plt.figure(figsize=(3.5, 3), dpi=1000)
ax = plt.subplot(222)
sn.set(font_scale=0.4)  # for label size
h_m = sn.heatmap(df_cm, cmap="BuPu", annot=True, annot_kws={"size": 5, "fontweight": "bold"}, fmt='d', cbar=True)
#
h_m.set_yticklabels(h_m.get_yticklabels(), rotation=0, fontweight="bold")
h_m.set_xticklabels(h_m.get_xticklabels(), rotation=45, fontweight="bold")
fig.savefig('fig' + '/' + 'confusion_matrix.png', dpi=1000)

#%%

acc = np.load("fig/acc.npy")
val_acc = np.load("fig/val_acc.npy")
loss = np.load("fig/loss.npy")
val_loss = np.load("fig/val_loss.npy")

#%%

plt.style.use(['science', 'ieee'])

#%%

with plt.style.context(['ieee']):
    fig = plt.figure(figsize=(3.5,3.5), dpi=1000)
    # 布局与图例
    layout = (2, 1)
    acc_ax = plt.subplot2grid(layout, (0, 0))
    loss_ax = plt.subplot2grid(layout, (1, 0))
    acc_ax.plot(acc, color='blue')
    acc_ax.plot(val_acc)
    acc_ax.set_title('Accuracy vs. Number of Training Epochs', fontweight="bold")
    acc_ax.set_ylabel('Accuracy', fontweight="bold")
    acc_ax.set_xlabel('Epochs', fontweight="bold")
    acc_ax.legend(['Training', 'Validation'])

    loss_ax.plot(loss, color='blue')
    loss_ax.plot(val_loss)
    loss_ax.set_title('Loss vs. Number of Training Epochs', fontweight="bold")
    loss_ax.set_ylabel('Loss', fontweight="bold")
    loss_ax.set_xlabel('Epochs', fontweight="bold")
    loss_ax.legend(['Training', 'Validation'])

    # 自动调整图例布局
    plt.tight_layout()
    plt.savefig('fig/acc+loss.png', dpi=1000)
    plt.show()

#%%



#%%



# accuracy的历史
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.grid()  # 生成网格
plt.savefig('fig/acc.png', dpi=1000)
plt.show()

#%%

# loss的历史
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'])
plt.grid()  # 生成网格
plt.savefig('fig/loss.png', dpi=1000)
plt.show()

#%%



#%%



#%%



#%%