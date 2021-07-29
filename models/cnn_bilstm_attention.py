from torch import nn
import torch
import torch.nn.functional as F
class TextCNN(nn.Module):
    def __init__(self, feaSize, contextSizeList, filterNum, name='textCNN'):
        super(TextCNN, self).__init__()
        self.name = name
        moduleList = []
        for i in range(len(contextSizeList)):
            moduleList.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=feaSize, out_channels=filterNum, kernel_size=contextSizeList[i], padding=contextSizeList[i]//2)
                    )
                )
        self.conv1dList = nn.ModuleList(moduleList)
        self.process=nn.Sequential(nn.BatchNorm1d(filterNum), nn.ReLU())
    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = x.transpose(1,2) # => batchSize × feaSize × seqLen
        x = [conv(x) for conv in self.conv1dList]  # => scaleNum * (batchSize × filterNum × seqLen)
        x_new=[self.process(new) for new in x]
        pooler=[torch.max(new,dim=2)[0] for new in x]
        pooling=[torch.relu(pool) for pool in pooler]
        return torch.cat(x_new, dim=1).transpose(1,2),torch.cat(pooling,dim=1) # => batchSize × seqLen × scaleNum*filterNum

class Simple_Protein_Predict(nn.Module):
    def __init__(self,lstm_hidden_size,lstm_hidden_size_1,contextSizeList,filter_number):
        super(Simple_Protein_Predict, self).__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_hidden_size_1 =lstm_hidden_size_1
        self.embed_matrix=nn.Embedding(22,64)
        self.lstm = nn.LSTM(len(contextSizeList)*filter_number,self.lstm_hidden_size,num_layers=2,batch_first=True,bidirectional=True)
        self.embed_matrix1 = nn.Embedding(22, 128)
        self.lstm1 = nn.LSTM(len(contextSizeList)*filter_number, self.lstm_hidden_size_1, num_layers=2,batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(.2)
        self.predict = nn.Linear(4 * self.lstm_hidden_size_1, 4 * self.lstm_hidden_size)
        self.linear=nn.Linear(self.lstm_hidden_size+self.lstm_hidden_size_1+2*filter_number*len(contextSizeList),2 * self.lstm_hidden_size)
        self.final_linear=nn.Linear(2 * self.lstm_hidden_size,1)
        self.activation = nn.Sigmoid()
        self.attention_layer1 = nn.Sequential(
            nn.Linear(self.lstm_hidden_size_1, self.lstm_hidden_size_1),
            nn.ReLU(inplace=True))
        self.attention_layer = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size),
            nn.ReLU(inplace=True))
        self.textCNN = TextCNN(64 , contextSizeList, filter_number)
        self.textCNN_second = TextCNN(128 , contextSizeList, filter_number)
    def exponent_neg_manhattan_distance(self, x1, x2):
        ''' Helper function for the similarity estimate of the LSTMs outputs '''
        return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))
    def attention_net_with_w(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer(lstm_hidden)
        # [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result
    def attention_net_with_w_virus(self, lstm_out, lstm_hidden):
        '''

        :param lstm_out:    [batch_size, len_seq, n_hidden * 2]
        :param lstm_hidden: [batch_size, num_layers * num_directions, n_hidden]
        :return: [batch_size, n_hidden]
        '''
        lstm_tmp_out = torch.chunk(lstm_out, 2, -1)
        # h [batch_size, time_step, hidden_dims]
        h = lstm_tmp_out[0] + lstm_tmp_out[1]
        # [batch_size, num_layers * num_directions, n_hidden]
        lstm_hidden = torch.sum(lstm_hidden, dim=1)
        # [batch_size, 1, n_hidden]
        lstm_hidden = lstm_hidden.unsqueeze(1)
        # atten_w [batch_size, 1, hidden_dims]
        atten_w = self.attention_layer1(lstm_hidden)
        # [batch_size, time_step, hidden_dims]
        m = nn.Tanh()(h)
        # atten_context [batch_size, 1, time_step]
        atten_context = torch.bmm(atten_w, m.transpose(1, 2))
        # softmax_w [batch_size, 1, time_step]
        softmax_w = F.softmax(atten_context, dim=-1)
        # context [batch_size, 1, hidden_dims]
        context = torch.bmm(softmax_w, h)
        result = context.squeeze(1)
        return result

    def forward(self, seq_1, seq_2, label):
        output1 = self.embed_matrix(seq_1)
        conv1, pooling_abs = self.textCNN(output1)
        orgin_output, output1 = self.lstm(conv1)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state = output1[0].permute(1, 0, 2)
        atten_out = self.attention_net_with_w(orgin_output, final_hidden_state)

        output2 = self.embed_matrix1(seq_2)
        conv2, pooling_virus = self.textCNN_second(output2)
        orgin_output1, output2= self.lstm1(conv2)
        # final_hidden_state : [batch_size, num_layers * num_directions, n_hidden]
        final_hidden_state1 = output2[0].permute(1, 0, 2)
        atten_out1 = self.attention_net_with_w_virus(orgin_output1, final_hidden_state1)
        predictions = self.linear(torch.cat((pooling_abs,atten_out,pooling_virus,atten_out1), 1))
        #predictions=self.exponent_neg_manhattan_distance(output1,output2).squeeze()
        predictions=self.final_linear(predictions)
        predictions=self.activation(predictions)
        predictions=predictions.squeeze()
        #predictions = self.activation(output).squeeze()
        # 截取#CLS#标签所对应的一条向量, 也就是时间序列维度(seq_len)的第0条

        # 下面是[batch_size, hidden_dim] 到 [batch_size, 1]的映射
        # 我们在这里要解决的是二分类问题
        # predictions = self.dense(first_token_tensor)
        # 用sigmoid函数做激活, 返回0-1之间的值
        #predictions = self.activation(outputs)
        compute_loss = nn.BCELoss()
        if label is not None:
            # 计算loss
            loss = compute_loss(predictions, label)
            return loss, predictions
        else:
            return predictions

