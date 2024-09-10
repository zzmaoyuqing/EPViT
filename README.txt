IMPORTANT NOTE: The dataset and ablation_dataset folders are missing from this directory due to upload data memory limitations, which can be downloaded Mendeley Data.

Code is provided to implement the EPViT:
Run the main.py script to train the model to get the results in the 'output/PPI5093', 'output/PPI3672', 'output/PPI2708' folder.
Note: The results of the ablation studies and comparative methods are in 'output' folder as well.


Optional plotting: The code for plotting bar charts, ROC and PR curves is in the 'plot' folder.
a).plot/ablation_PPI_sub:
1. The plot_bar.ipynb script plots bar charts for different omics data on three PPI datasets.
2. The plot_ROC_PR.py script plots ROC and PR curves of the ablation experiments on three PPI datasets.

b).plot/comparision_centrality:
1.The plot_bar.ipynb script plots bar charts for multi-feature fusion methods on three PPI datasets.

c).plot/comparision_deep_learning:
1.The plot_bar.ipynb script plots bar charts for deep learning-based methods on three PPI datasets.
2. The plot_ROC_PR.py script plots ROC and PR curves for deep learning-based methods on three PPI datasets.

d).plot/loss_function:
1. The plot_bar_diff_lossfunc.ipynb script plots bar charts for focal loss function and cross entropy loss function on three PPI datasets.
2. The plot_ROC_PR.py script plots ROC and PR curves on three PPI datasets.

environment:
python==3.8.16
torch==2.0.1
einops==0.6.1
networkx==3.1
