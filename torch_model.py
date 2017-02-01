import torch.nn as nn

class PGModel(nn.Module):
    def __init__(self, data_size, hidden_size, output_size, stdv):
        super(PGModel, self).__init__()
        self.linear = nn.Linear(data_size, hidden_size)
        self.tanh=nn.Tanh()
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.linear.weight.data.normal_(0, stdv)
        self.linear.bias.data.normal_(0, stdv)
        self.linear2.weight.data.normal_(0, stdv)
        self.linear2.bias.data.normal_(0, stdv)
        self.data_size = data_size

    def forward(self, data):
        output = self.linear(data)
        output = self.tanh(output)
        output = self.linear2(output)
        output = self.softmax(output)
        return output
