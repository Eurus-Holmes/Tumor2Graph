import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, feaSize, filterNum, cnn_kernels):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=feaSize, out_channels=filterNum[0], kernel_size=cnn_kernels[0],
                      padding=cnn_kernels[0] // 2),
            nn.BatchNorm1d(filterNum[0]),
            nn.ReLU()
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=filterNum[0], out_channels=filterNum[1], kernel_size=cnn_kernels[1],
                      padding=cnn_kernels[1] // 2),
            nn.BatchNorm1d(filterNum[1]),
            nn.ReLU()
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=filterNum[1], out_channels=filterNum[2], kernel_size=cnn_kernels[2],
                      padding=cnn_kernels[2] // 2),
            nn.BatchNorm1d(filterNum[2]),
            nn.ReLU(),
            )

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1, 2)  # 维度转换=> batchSize × feaSize × seqLen
        #print(x.size())
        out1 = self.layer1(x)
        #print(out1.size())
        out2 = self.layer2(out1)
        #print(out2.size())
        out3 = self.layer3(out2)
        #print(out3.size())
        #exit()
        # print(out1.size())
        out3, d = out3.max(dim=-1)
        out3 = out3.view(out3.size()[0], -1)
        return out3


class Simple_Protein_Predict(nn.Module):
    def __init__(self, protein_ft_dim):
        super(Simple_Protein_Predict, self).__init__()
        self.filter_anti = [protein_ft_dim, 64, 96]
        self.filter_virus = [protein_ft_dim, 96, 96]
        self.cnn_anti_kernels = [5, 7, 9]
        self.cnn_virus_kernels = [5, 7, 13]
        # self.embed_matrix1 = nn.Embedding(22, 64)
        # self.embed_matrix2 = nn.Embedding(22, 128)

        self.ConvNet_seq1 = ConvNet(64, self.filter_anti, self.cnn_anti_kernels)
        self.ConvNet_seq2 = ConvNet(128, self.filter_virus, self.cnn_virus_kernels)
        self.dropout = nn.Dropout(.2)
        self.activation = nn.ELU()

        # self.linear1 = nn.Linear(self.filter_anti[-1] + self.filter_virus[-1], 64)
        self.linear1 = nn.Linear(self.filter_anti[-1], 96)  # 点乘
        self.linear2 = nn.Linear(96, 96)
        self.linear3 = nn.Linear(96, 1)

    def forward(self, seq_1, seq_2, label):
        output1 = self.embed_matrix1(seq_1)
        conv1 = self.ConvNet_seq1(output1)

        output2 = self.embed_matrix2(seq_2)
        conv2 = self.ConvNet_seq2(output2)

        # print(conv1.size(), conv2.size())
        # exit()
        # 拼接
        # flag = torch.cat((conv1, conv2), 1)
        # 点乘
        flag = (conv1 + conv2) + (conv1 * conv2)
        # print(flag.size())
        # exit()
        # 3层MLP
        predictions = self.linear1(flag)
        predictions = self.activation(predictions)
        predictions_1 = self.dropout(predictions)

        predictions_1 = self.linear2(predictions_1)
        predictions_1 = self.activation(predictions_1)
        predictions_2 = self.dropout(predictions_1)

        predictions_2 = self.linear2(predictions_2)
        predictions_2 = self.activation(predictions_2)
        predictions_3 = self.dropout(predictions_2)

        predictions = flag + predictions_1 + predictions_2 + predictions_3

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