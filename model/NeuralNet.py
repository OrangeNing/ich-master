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
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet(nn.Module):
    def __init__(self, embed_size=2048 * 3, LSTM_UNITS=64, DO=0.3):
        super(NeuralNet, self).__init__()

        self.embedding_dropout = SpatialDropout(DO)  # DO)   0.0

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        # self.linear3 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        # self.linear4 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear = nn.Linear(LSTM_UNITS * 2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        h_embadd = torch.cat((h_embedding[:, :, :2048], h_embedding[:, :, :2048]), -1)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        # h_lstm3, _ = self.lstm1(h_lstm2)
        # h_lstm4, _ = self.lstm2(h_lstm3)
        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))
        # h_conc_linear3 = F.relu(self.linear1(h_lstm3))
        # h_conc_linear4 = F.relu(self.linear2(h_lstm4))
        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2 + h_embadd

        output = self.linear(hidden)

        return output

