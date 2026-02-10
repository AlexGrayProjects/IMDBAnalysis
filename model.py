<<<<<<< Updated upstream
import torch
from torch import nn
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Reviews(torch.utils.data.Dataset):
    def __init__(self, df):
        self.reviews = df['tokenizedReview'].tolist()
        self.labels = df['sentiment'].tolist()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, i):
        return self.reviews[i], float(self.labels[i])


class LSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size, embedded_size=None):
        if embedded_size == None:
            embedded_size = hidden_size
        self.hidden_size = hidden_size
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=0)
        self.model_y = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.forget = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.input_gate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.candidate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        
        self.tanh = nn.Tanh()
        
        nn.init.constant_(self.forget.bias, 1.0)
        
    def forward(self, tokens):
        batch_size, seq_length = tokens.shape
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        for i in range(0, seq_length):
            x = tokens[:, i]
            mask = (x != 0).unsqueeze(1).float()
            h_new, c_new = self.lstm_block(x, h, c)
            h_new, c_new = self.lstm_block(x, h, c)
            h = mask * h_new + (1 - mask) * h
            c = mask * c_new + (1 - mask) * c
        h = self.dropout(h)
        y = self.model_y(h)
        return torch.sigmoid(y)

    def lstm_block(self, x, hprev, cprev):
        embedded = torch.cat([self.embedding(x), hprev], dim=1)
        f = torch.sigmoid(self.forget(embedded))
        i = torch.sigmoid(self.input_gate(embedded))
        candidate_c = self.tanh(self.candidate(embedded))    
        c = torch.mul(f, cprev) + torch.mul(i, candidate_c)
        output = torch.sigmoid(self.output_gate(embedded))
        h = torch.mul(output, self.tanh(c))
=======
import torch
from torch import nn
from torch.utils.data import DataLoader
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Reviews(torch.utils.data.Dataset):
    def __init__(self, df):
        self.reviews = df['tokenizedReview'].tolist()
        self.labels = df['sentiment'].tolist()

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, i):
        return self.reviews[i], float(self.labels[i])


class LSTM(nn.Module):

    def __init__(self, vocab_size, hidden_size, embedded_size=None):
        if embedded_size == None:
            embedded_size = hidden_size
        self.hidden_size = hidden_size
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size, padding_idx=0)
        self.model_y = nn.Linear(self.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        
        self.forget = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.input_gate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.candidate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        self.output_gate = nn.Linear(embedded_size+self.hidden_size, self.hidden_size)
        
        self.tanh = nn.Tanh()
        
        nn.init.constant_(self.forget.bias, 1.0)
        
    def forward(self, tokens):
        batch_size, seq_length = tokens.shape
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        for i in range(0, seq_length):
            x = tokens[:, i]
            mask = (x != 0).unsqueeze(1).float()
            h_new, c_new = self.lstm_block(x, h, c)
            h_new, c_new = self.lstm_block(x, h, c)
            h = mask * h_new + (1 - mask) * h
            c = mask * c_new + (1 - mask) * c
        h = self.dropout(h)
        y = self.model_y(h)
        return torch.sigmoid(y)

    def lstm_block(self, x, hprev, cprev):
        embedded = torch.cat([self.embedding(x), hprev], dim=1)
        f = torch.sigmoid(self.forget(embedded))
        i = torch.sigmoid(self.input_gate(embedded))
        candidate_c = self.tanh(self.candidate(embedded))    
        c = torch.mul(f, cprev) + torch.mul(i, candidate_c)
        output = torch.sigmoid(self.output_gate(embedded))
        h = torch.mul(output, self.tanh(c))
>>>>>>> Stashed changes
        return h, c