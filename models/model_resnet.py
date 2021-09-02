import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import Parameter


class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        nn.ELU(),
                        nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                        nn.ELU(),
                        nn.Dropout(dropout),
                        nn.Conv1d(filterSize, filterSize, kernel_size=3, padding=dilaSize, dilation=dilaSize),
                        nn.InstanceNorm1d(filterSize),
                    )
        self.name = name

    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)


class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1, 2, 4, 8, 16], dropout=0.15, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize, filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i % len(dilaSizeList)], filterSize, dropout=dropout))
        self.name = name

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x)  # => batchSize × seqLen × filterSize
        # print("x->", x.size())
        x = self.blockLayers(x.transpose(1, 2))  # => batchSize × filterSize × seqLen
        # print("x-->", x.size())
        return F.elu(x)  # => batchSize × filterSize × seqLen


class Simple_Protein_Predict(nn.Module):
    def __init__(self):
        super(Simple_Protein_Predict, self).__init__()
        self.embed_matrix1 = nn.Embedding(21, 64)
        self.embed_matrix2 = nn.Embedding(21, 128)

        self.ResDilaCNNBlocks_seq1 = ResDilaCNNBlocks(feaSize=64, filterSize=64)
        self.ResDilaCNNBlocks_seq2 = ResDilaCNNBlocks(feaSize=128, filterSize=64)
        self.dropout = nn.Dropout(.2)
        self.activation = nn.ELU()

        self.linear1 = nn.Linear(64, 64)
        # self.linear1 = nn.Linear(self.filter_anti[-1], 64) # 点乘
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, seq_1, seq_2, label):
        output1 = self.embed_matrix1(seq_1)
        conv1 = self.ResDilaCNNBlocks_seq1(output1)
        # print(conv1.size())
        conv1, d = conv1.max(dim=-1)
        # print(conv1.size())
        conv1 = conv1.view(conv1.size()[0], -1)
        # print(conv1.size())
        # exit()
        output2 = self.embed_matrix2(seq_2)
        # print('output2->', output2.size())
        conv2 = self.ResDilaCNNBlocks_seq2(output2)
        conv2, d = conv2.max(dim=-1)
        conv2 = conv2.view(conv2.size()[0], -1)
        # print('seq_1->', conv1.size())
        # print('seq_2->', conv2.size())
        # 点乘
        flag = (conv1 + conv2) + (conv1 * conv2)
        # print(flag.size())
        # exit()
        # 3层MLP
        predictions = self.linear1(flag)
        predictions = self.activation(predictions)
        predictions = self.dropout(predictions)

        predictions = self.linear2(predictions)
        predictions = self.activation(predictions)
        predictions = self.dropout(predictions)

        predictions = self.linear3(predictions)

        predictions = torch.sigmoid(predictions)
        predictions = predictions.squeeze()
        compute_loss = nn.BCELoss()
        if label is not None:
            # 计算loss
            loss = compute_loss(predictions, label)
            return loss, predictions
        else:
            return predictions
