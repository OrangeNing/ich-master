import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
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
    def __init__(self, embed_size=6144, LSTM_UNITS=64, DO=0.6):
        super(BiLSTM, self).__init__()
        self.embedding_dropout = SpatialDropout(DO)  # DO)
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, n_classes)
        self.linear_slice_fc1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear_slice_fc2 = nn.Linear(LSTM_UNITS * 2, n_classes)
    def forward(self, x, lengths=None):
        h_embedding = x
        output1, (h_1,c_1) = self.lstm1(h_embedding)
        output2, (h_2, c_2) = self.lstm2(output1)
        # output1 = F.relu(self.linear1(output1))
        # output2 = F.relu(self.linear2(output2))
        out_case = torch.sum(output2,dim = 1)
        out_case  = self.linear1(out_case)
        out_case = self.linear2(out_case)
        output_slice =  output2

        output_slice = self.linear_slice_fc1(output_slice)
        output_slice = self.linear_slice_fc2(output_slice)

        return out_case,output_slice


class NeuralNet(nn.Module):
    def __init__(self, embed_size=8192, LSTM_UNITS=64, DO=0.3):
        super(NeuralNet, self).__init__()

        self.embedding_dropout = SpatialDropout(DO)  # DO)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear = nn.Linear(LSTM_UNITS * 2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        # h_embadd = torch.cat((h_embedding[:, :, :2048], h_embedding[:, :, :2048]), -1)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))

        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear(hidden)

        return output


class GRU(nn.Module):
    def __init__(self, embed_size=8192, LSTM_UNITS=64, DO=0.3):
        super(GRU, self).__init__()
        self.gru = nn.GRU(embed_size, LSTM_UNITS, batch_first=True)
        # self.embedding_dropout = SpatialDropout(DO)  # DO)   0.0
        self.linear = nn.Linear(2048, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x
        out,_ = self.gru(h_embedding)
        output = self.linear(out)
        return output
# https://www.kaggle.com/bminixhofer/speed-up-your-rnn-with-sequence-bucketing
class NeuralNet_v2(nn.Module):
    def __init__(self, embed_size=8192, LSTM_UNITS=64, DO=0.3):
        super(NeuralNet_v2, self).__init__()

        self.embedding_dropout = SpatialDropout(0.0)  # DO)

        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)
        self.linear2 = nn.Linear(LSTM_UNITS * 2, LSTM_UNITS * 2)

        self.linear = nn.Linear(LSTM_UNITS * 2, n_classes)

    def forward(self, x, lengths=None):
        h_embedding = x

        # h_embadd = torch.cat((h_embedding[:, :, :2048], h_embedding[:, :, :2048]), -1)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)

        h_conc_linear1 = F.relu(self.linear1(h_lstm1))
        h_conc_linear2 = F.relu(self.linear2(h_lstm2))

        hidden = h_lstm1 + h_lstm2 + h_conc_linear1 + h_conc_linear2

        output = self.linear(hidden)

        return output


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_dim, bias=True):
        """
        Initialize LSTM cell.

        Parameters
        ----------
        input_size: int
        hidden_dim: int
            Number of channels of hidden state.
        bias: bool
            Whether or not to add the bias.



        """

        super(LSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.Gates = nn.Linear(self.input_size + self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        stacked_inputs = torch.cat([input_tensor, h_cur], dim=1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * c_cur) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())

class NestedLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_dim, bias=True):
        super(NestedLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.Gates = nn.Linear(self.input_size + self.hidden_dim, 4 * self.hidden_dim, bias=self.bias)
        self.cell_core = LSTMCell(2 * self.hidden_dim, self.hidden_dim, self.bias)

    def forward(self, input_tensor, total_state):
        cur_state, cell_state = total_state
        h_cur, c_cur = cur_state
        # print(input_tensor.size())
        # print(h_cur.size())
        if input_tensor.size()[0] != h_cur.size()[0]:
            if input_tensor.size()[0]>h_cur.size()[0]:
                input_tensor = input_tensor[:h_cur.size()[0]]
            else:
                h_cur= h_cur[:input_tensor.size()[0]]
                c_cur = c_cur[:input_tensor.size()[0]]
        # else:
        #     print("equal")
        stacked_inputs = torch.cat((input_tensor, h_cur), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = torch.cat(((remember_gate * c_cur), (in_gate * cell_gate)), 1)
        #         print(cell.size())

        cell_state = self.cell_core(cell, cell_state)
        cell = cell_state[0]
        hidden = out_gate * torch.tanh(cell)

        return ((hidden, cell), cell_state)

    def init_hidden(self, batch_size):
        cell_state = (Variable(torch.zeros(batch_size, self.hidden_dim)).cuda(),
                      Variable(torch.zeros(batch_size, self.hidden_dim)).cuda())

        return (cell_state, self.cell_core.init_hidden(batch_size))


class NestedLSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers=1,
                 batch_first=True, bias=True, return_all_layers=False):

        super(NestedLSTM, self).__init__()

        # Make sure that  `hidden_dim` are lists having len == num_layers
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_size = input_size

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cell_list.append(NestedLSTMCell(input_size=self.input_size,
                                            hidden_dim=self.hidden_dim[i],
                                            bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)###???
        self.linear1 = nn.Linear(2048 , 2048)
        self.linear2 = nn.Linear(6144, 2048)
        self.linear = nn.Linear(2048, n_classes) #dense
    def forward(self, input_tensor, hidden_state =None):
    # def forward(self, input_tensor):
        """

        Parameters
        ----------
        input_tensor: todo
            3-D Tensor either of shape (t, b, c) or (b, t, c)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        h_embedding = input_tensor

        h_embadd = torch.cat((h_embedding[:, :, :2048], h_embedding[:, :, :2048]), -1)
        if not self.batch_first:
            # (t, b, c) -> (b, t, c)
            input_tensor = input_tensor.permute(1, 0, 2)
        # print(input_tensor[0][0])
        if hidden_state == None:
            bs = input_tensor.size()[0]
            # print(bs)
            hidden_state = self.init_hidden(bs)
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            state = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                state = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :],
                                                  total_state=state)
                output_inner.append(state[0][0])

            layer_output = torch.stack(output_inner, dim=1)

            cur_layer_input = layer_output
            if not self.batch_first:
                layer_output_list.append(layer_output.permute(1, 0, 2))
            else:
                layer_output_list.append(layer_output)
            last_state_list.append(state)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        # layer_output = torch.sigmoid(layer_output)

        # h_embadd = self.linear2(h_embadd)
        h_conc_linear1 = F.relu(self.linear1(layer_output))
        out = layer_output + h_conc_linear1
        output = self.linear(out)

        # output = self.linear(layer_output_list)
        return output
        # return layer_output_list, last_state_list, hidden_state

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param