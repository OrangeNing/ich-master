import torch
from torch import nn
from torch.nn import functional as F
n_classes=6
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
class BiLSTM(nn.Module):
    # def __init__(self, embed_size=trnemb.shape[-1] * 3, LSTM_UNITS=64, DO=0.3):
    def __init__(self, embed_size=8192, LSTM_UNITS=64, DO=0.3):
        super(BiLSTM, self).__init__()

        self.embedding_dropout = SpatialDropout(0.0)  # DO)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)#bidirectional
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        ##
        self.lstm3 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear3 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        # self.lstm4 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        # self.linear4 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        ##
        ##attation
        # self.dense_layer1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        # self.dense_layer2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        #
        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear = nn.Linear(LSTM_UNITS * 2, n_classes)#dense

    # def attention(self,lstm_output,final_state):
    #     hidden = final_state.view(-1,LSTM_UNITS*2,1)
    #     attention_weights = torch.bmm(lstm_output,hidden)
    #     soft_attention_weights = F.softmax(attention_weights,1)
    #     context = torch.bmm(lstm_output.transpose(1,2),soft_attention_weights.unsqueeze(2))
    #     return context,soft_attention_weights

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:, :, :2048], h_embedding[:, :, :2048]), -1)
        # input = h_embedding.permute(1,0,2)
        # h_lstm1,(final_hidden_state,final_cell_state) = self.lstm1(input)
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        h_lstm3, _ = self.lstm3(h_lstm2)
        # h_lstm4, _ = self.lstm4(h_lstm3)
        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))
        h_conc_linear3 = F.relu(self.linear3(h_lstm3))
        # h_conc_linear4 = F.relu(self.linear4(h_lstm4))
        hidden = h_lstm1 + h_lstm2 + h_lstm3  + h_conc_linear1 + h_conc_linear2 + h_conc_linear3 + h_embadd
        # hidden,attention = self.attention(h_lstm1,final_hidden_state)
        output = self.linear(hidden)

        return output