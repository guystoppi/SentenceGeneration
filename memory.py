from torch import nn

class MemoryLSTM(nn.Module):

    def __init__(self, input_size, out_size, nlayers=4, hiddensize=512, dropout=0.4, reset=None):
        super(MemoryLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hiddensize, num_layers=nlayers, 
                            dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.lin = nn.Linear(hiddensize, out_size)

        self.hidden_state = None
        self.reset_freq = reset
        self.step_count = 0

    def step(self):
        self.step_count += 1

        if self.hidden_state:
            self.hidden_state = tuple(tsr.detach() for tsr in self.hidden_state)

        if self.reset_freq and self.step_count % self.reset == 0:
            self.reset()

    def reset(self):
        self.hidden_state = None

    def forward(self, embeds):
        # B x N x V
        if self.hidden_state:
            out, self.hidden_state = self.lstm(embeds, self.hidden_state)
        else:
            out, self.hidden_state = self.lstm(embeds)
        
        # B x N x V
        out = self.drop(out) # B x N x V
        out = self.lin(out) # B x N x V

        return out
