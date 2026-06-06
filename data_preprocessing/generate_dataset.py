import argparse
import os
import numpy as np
from node2vec import node2vec
import GeMyData
import networkx as nx
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='PPI2708')  # choose PPI5093 or PPI3672 or PPI2708
    parser.add_argument('--omics_data', type=str, default='PPI+sub')  # choose 'PPI+sub' for EPViT, ablation experiments with 'PPI_only' or 'sub_only'

    parser.add_argument('--p', type=float, default=1, help='return parameter')
    parser.add_argument('--q', type=float, default=0.5, help='in-out parameter')
    parser.add_argument('--d', type=int, default=224, help='dimension')
    parser.add_argument('--r', type=int, default=10, help='walks per node')
    parser.add_argument('--l', type=int, default=80, help='walk length')
    parser.add_argument('--k', type=float, default=10, help='window size')
    args = parser.parse_args()

    return args

def features_extraction_from_PPI(args, ppi_file, ppi_target_file):
    '''Get result of node2vec function. Ensure that nodes and labels correspond to each other.
    '''

    PPI_graph = nx.read_edgelist(ppi_file)
    G = nx.Graph()
    G.add_nodes_from(sorted(PPI_graph.nodes(data=True)))
    G.add_edges_from(PPI_graph.edges(data=True))

    # Add the weight attribute to the edge of the graph.
    for u, v in G.edges:
        G.add_edge(u, v, weight=1)

    # Add the target attribute to the node in the graph.
    data = pd.read_csv(ppi_target_file)
    target = data.iloc[:, 2]
    target_mapping = dict(zip(G.nodes, target))
    for node in G.nodes:
        G.nodes[node]['y_target'] = target_mapping[node]

    vec = node2vec(args, G)

    nodes, walks = vec.get_walks()
    #print("the shape of walks: ", np.array(walks).shape)  # (50930,80)

    embeddings = vec.learning_features(nodes, walks)            # Get the embedding for each node
    targets = [list(G.nodes[node].values()) for node in nodes]  # Get the label of each embedding corresponding to its node
    #print('the shape of embeddings: ', np.array(embeddings).shape) # (5093,224)

    return embeddings, targets, nodes


def get_ascending_nodes(embeddings, targets, nodes):
    '''The order is in ascending order of protein names.
    '''

    df = pd.DataFrame(embeddings, index=np.array(nodes).reshape(-1, 1))
    # df.insert(0, 'protein', protein)
    df.insert(0, 'target', targets)
    df_sorted = df.sort_index()
    sorted_protein_emd = df_sorted.values[:, 1:]

    return sorted_protein_emd


def normalized_pro_emb(sorted_protein_emd):
    '''Normalize the embedding corresponding to the ascending sorted proteins to (0, 1).
    '''

    Standard_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(sorted_protein_emd.T)
    sorted_protein_emd_norm = Standard_data.T

    return sorted_protein_emd_norm



def normalized_subcellular(sub_data):
    '''Processing subcellular localization data. Normalizing the CONFIDENCE column.
    '''

    values = sub_data['confidence'].values.reshape(-1, 1)
    normalized_data = MinMaxScaler(feature_range=(0, 1)).fit_transform(values)

    # Since the normalized data is in array format, it is converted to a data frame
    normalized_data = pd.DataFrame(normalized_data)

    # A new column is added to sub_data, which is the result of normalizing the value of confidence
    sub_data['normalized confidence'] = normalized_data

    return sub_data



def get_PPI_sub_matrix(sub_data, sub_location, PPI_file):
    '''Get matrix S_S with proteins in the rows and subcelullar locations in the columns.
    '''
    sub_pro = list(sub_data.iloc[:, 0])
    sub_number = len(sub_pro)

    # read PPI file
    with open(PPI_file, 'r') as file:
        lines = file.readlines()

    # compute nodes in PPI network
    Pro = []
    for line in lines:
        proteins = line.strip().split()
        Pro.extend(proteins)

    # Remove duplicates and get the total number of proteins
    Pro = np.unique(Pro)
    Pro = np.insert(Pro, 0, 'pro')  # insert one row to Pro
    number = len(Pro) - 1  # number of nodes in PPI

    # Find indices of yeast_sub proteins in Pro.
    # The edgef stores the indexes, and these index values represent the indexes of the proteins in the subcellular localization file in Pro
    edgef = np.zeros((sub_number, 1), dtype=int)
    for i in range(sub_number):
        protein = sub_pro[i]
        if protein in Pro:
            # Pro is one-dimension array, return index array by np.where function.
            index = np.where(Pro == protein)[0][0]
            edgef[i, 0] = index

    PPI_sub_data = np.zeros((number, len(sub_location)))
    for i in range(sub_number):
        if edgef[i] != 0:
            # Stored are indexes, and these index values represent the index of the subcell location in the sub_location in the subcell location file
            index2 = np.where(sub_location == sub_data.loc[i]['sub'])[0][0]  # Need to choose your own column
            PPI_sub_data[edgef[i]-1, index2] = sub_data.loc[i]['normalized confidence']

    # Calculate the sum of each line and insert into the first column of each row
    row_sums = np.sum(PPI_sub_data, axis=1)
    PPI_sub_data = np.insert(PPI_sub_data, 0, row_sums, axis=1)

    return PPI_sub_data


def vec2matrix(args, pro_emb, sub_emb):
    '''
    Fusion of PPI and subcellular localization data
    '''
    a = []
    for i in range(len(pro_emb)):
        emb_sub = np.dot(pro_emb[i].reshape(args.d, 1), sub_emb[i].reshape(1, args.d))
        a.append(emb_sub)
    return a



args = args_parser()

# input file
ppi_file = 'data/' + args.network + '/yeast_' + args.network + '.txt'
ppi_target_file = 'data/' + args.network + '/protein_target.txt'
sub_file = 'data/yeast_compartment_integrated_full.tsv'

# save output file in dataset folder
if args.omics_data == 'PPI+sub':
    train_data_npy_file = '../dataset/' + args.network + '/dim224/train_test/train_set/train_data.npy'
    train_target_npy_file = '../dataset/' + args.network + '/dim224/train_test/train_set/train_target.npy'
    test_data_npy_file = '../dataset/' + args.network + '/dim224/train_test/test_set/test_data.npy'
    test_target_npy_file = '../dataset/' + args.network + '/dim224/train_test/test_set/test_target.npy'
elif args.omics_data == 'PPI_only':
    train_data_npy_file = '../ablation_dataset/ablation_PPI_only/' + args.network + '/dim224/train_test/train_set/train_data.npy'
    train_target_npy_file = '../ablation_dataset/ablation_PPI_only/' + args.network + '/dim224/train_test/train_set/train_target.npy'
    test_data_npy_file = '../ablation_dataset/ablation_PPI_only/' + args.network + '/dim224/train_test/test_set/test_data.npy'
    test_target_npy_file = '../ablation_dataset/ablation_PPI_only/' + args.network + '/dim224/train_test/test_set/test_target.npy'
else:
    train_data_npy_file = '../ablation_dataset/ablation_sub_only/' + args.network + '/dim224/train_test/train_set/train_data.npy'
    train_target_npy_file = '../ablation_dataset/ablation_sub_only/' + args.network + '/dim224/train_test/train_set/train_target.npy'
    test_data_npy_file = '../ablation_dataset/ablation_sub_only/' + args.network + '/dim224/train_test/test_set/test_data.npy'
    test_target_npy_file = '../ablation_dataset/ablation_sub_only/' + args.network + '/dim224/train_test/test_set/test_target.npy'



# Features extraction from PPI data
embeddings, targets, nodes = features_extraction_from_PPI(args, ppi_file, ppi_target_file)
# Get the proteins in ascending order and normalize the values
sorted_protein_emd = get_ascending_nodes(embeddings, targets, nodes)
sorted_protein_emd_norm = normalized_pro_emb(sorted_protein_emd)


# Features extraction from subcellular localization data
# Save column 'standard pro', 'sub', 'confidence'. According to your file, you can choose your interested subcellular locations
sub_data = pd.read_csv(sub_file, sep='\t', names=['pro', 'standard pro', 'go', 'sub', 'confidence'], header=None)
sub_data.drop(['standard pro', 'go'], axis=1, inplace=True)
# Get subcellular locations
sub_location = sub_data['sub'].unique()  # 2702
# Normalized column 'confidence'
normalized_sub_data = normalized_subcellular(sub_data)
#normalized_sub_data.to_csv('normalized_sub_data.csv', index=False)

# Obtain the corresponding subcellular localizations of nodes in PPI
PPI_sub_matrix = get_PPI_sub_matrix(normalized_sub_data, sub_location, ppi_file)  # (5093,2703), (3672,2703), (2708,2703)
sorted_PPI_sub_emb_norm = PPI_sub_matrix[:, 1:args.d + 1]                         # (5093,224), (3672,224), (2708,224)


# Fusion of PPI and subcellular localization data
# Features fusion by outer product operation
if args.omics_data == 'PPI+sub':
    dot_emb_sub = vec2matrix(args, sorted_protein_emd_norm, sorted_PPI_sub_emb_norm)  # ppi+sub
elif args.omics_data == 'PPI_only':
    dot_emb_sub = vec2matrix(args, sorted_protein_emd_norm, sorted_protein_emd_norm)  # only ppi
else:
    dot_emb_sub = vec2matrix(args, sorted_PPI_sub_emb_norm, sorted_PPI_sub_emb_norm)  # only sub

# Generate dataset
dataset = GeMyData.GeMyDataset(dot_emb_sub, ppi_target_file)

# Divide 80% of the training set and 20% of the testing set
train_dataset, test_dataset = GeMyData.split_data2(dataset)

# Read the divided train and test dataset, respectively.
train_data = GeMyData.read_split_data(dataset, train_dataset)
test_data = GeMyData.read_split_data(dataset, test_dataset)

# Read the divided train_target and test_target labels, respectively.
train_target = GeMyData.read_split_target(dataset, train_dataset)
test_target = GeMyData.read_split_target(dataset, test_dataset)

train_es_num = int(train_target.sum())  # 926   train_data=4075
test_es_num = int(test_target.sum()) # 241   test_data=1018

print('train_es_num:', train_es_num, "\ntest_es_num:", test_es_num)

'''
    -------------------------------PPI5093-----------------------------------------------
            train_es_num = train_target.sum()  # 926   train_data=4075
            test_es_num = test_target.sum()    # 241   test_data=1018

    -------------------------------PPI3672-----------------------------------------------
            train_es_num = train_target.sum()  # 756   train_data=2938
            test_es_num = test_target.sum()    # 172   test_data=734

    -------------------------------PPI2708-----------------------------------------------
            train_es_num = train_target.sum()  # 622   train_data=2167
            test_es_num = test_target.sum()    # 163   test_data=541
'''

# save processed dataset
# np.save(train_data_npy_file, train_data)
# np.save(train_target_npy_file, train_target)
# np.save(test_data_npy_file, test_data)
# np.save(test_target_npy_file, test_target)