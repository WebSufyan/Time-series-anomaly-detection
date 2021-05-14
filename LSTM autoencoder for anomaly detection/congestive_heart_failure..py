#scipy library to load arff file
from scipy.io import arff

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdm

device = torch.device('cuda') # this is just for running our model on the GPU for faster perfoemance and computing

''' ################# loading data ########################################'''
#loading train and test 
train = arff.loadarff('data/ECG5000_TRAIN.arff')
test = arff.loadarff('data/ECG5000_TEST.arff')

#transform arff data file to pandas dataframe
train = pd.DataFrame(train[0])
test = pd.DataFrame(test[0])

#concatenate train with test
dataset = pd.concat((train, test), axis=0)

#count how many samples we have for each class
classes = dataset.target.value_counts()


#check how many examples for each heartbeat class do we have
# sns.countplot(x = 'target', data = dataset) 

#b'1' is considered normal and other classes will be considered anomalies, we will deal with them as anomalies


''' ################# proccessing data ########################################'''

# split to train and test
train, test = train_test_split(dataset, test_size = 0.05, shuffle = True, random_state = 101)

# get all normal heartbeats and drop target
train_normal_df = train[train.target == b'1'].drop(labels='target', axis=1) # this is the data that we will train our autoencoder on.

# get all abnormal heartbeats from training set
train_abnormal_df = train[train.target != b'1'].drop(labels='target', axis=1)

# same thing for the test test
test_normal_df = test[test.target == b'1'].drop(labels='target', axis=1)
test_abnormal_df = test[test.target != b'1'].drop(labels='target', axis=1)


# create a small custom dataset that takes in our dataset and turn it into 2D tensor (140,1) this will be useful for latr on when testing
class dataset():
    def __init__(self, dataset):
        self.dataset = dataset.values
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = torch.tensor(self.dataset[idx]).unsqueeze(1).float()
        
        return data
        
normal_train = dataset(train_normal_df)

#load our dataset in batches for the model, our input data will be in shape of (batch size, sequence lenght, number of features)
train_data = DataLoader(normal_train, batch_size=64, shuffle = True) # (64, 140, 1)



''' ################# GRU Autoencoder  ########################################'''

class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super(Encoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 2
        
        self.GRU1 = nn.GRU(n_features, self.hidden_dim, num_layers=1, batch_first=True)
        self.GRU2 = nn.GRU(self.hidden_dim, embedding_dim, num_layers=1, batch_first=True)
        
    def forward(self, x):
        
        x, hidden = self.GRU1(x)
        out, hidden = self.GRU2(x)
        
        return out
        

class Decoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = embedding_dim * 2
        
        self.GRU1 = nn.GRU(embedding_dim, embedding_dim, num_layers=1, batch_first=True)
        self.GRU2 = nn.GRU(embedding_dim, self.hidden_dim, num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_dim, n_features)
        
    def forward(self, x):

        x, hidden = self.GRU1(x)
        x, hidden = self.GRU2(x)

        
        return self.fc(x)

class GRUAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=32):
        super(GRUAutoencoder, self).__init__()
        
        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, n_features, embedding_dim).to(device)
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x


''' ################# Training our model  ########################################'''
torch.manual_seed(101)

model = GRUAutoencoder(140, 1).to(device)

criterion = nn.L1Loss(reduction='sum') # L1 loss funstion is the sum of the absolute of all differences between the true value and the predicted value

optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005) # Adam optimizer

epochs = 140

losses = []
model.train() # training mode
for epoch in tqdm(range(epochs)):
    
    for idx, i in enumerate(train_data):
        
        y_true = i.to(device)
        y_pred = model(y_true)
        
        loss = criterion(y_pred, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss)
    if epoch % 20 == 0:    
        print(loss)
    
# visualizing the losses

plt.plot(losses)
# sns.distplot(losses, bins=50, kde=True)


''' ################# testing the model  ########################################'''

# encoding normal class as 1 and the rest as 0
class_dict = {b'1' : 1, b'2' : 0, b'3' : 0, b'4' : 0, b'5' : 0}
test['target'] = test['target'].apply(lambda x : class_dict[x])

x_test = test.drop(labels = 'target', axis=1)
y_test = test['target']

# choosing a theshold for prediction
threshold = 2.60

model.eval() #evaluation mode

test_train = dataset(x_test)

correct = []
for j in test_train:
    
    y_real = j.unsqueeze(0).to(device)
    
    with torch.no_grad():
        y_val = model(y_real).to(device)
        
        cost = criterion(y_val, y_real) # using the loss for our prediction

        correct.append(cost)
        
        
is_it_correct = sum(l < threshold for l in correct) # applying threshold to capture how many samples our model will consider normal

normal_count = len(test[test['target'] == 1]) # in test set we have 140 normal samples.

sns.distplot(correct)

print(f'the model considered {is_it_correct.item()} samples from test set as normal \
      and the total of actual normal samples in our test set is {normal_count} sample \
      \nthe accuracy on test set is {(is_it_correct.item()/normal_count)*100:.2f} %')

"***************** model accuracy 97% *****************"

''' saving the model '''

# torch.save(model.state_dict(), 'GRUautoencoder.pt')











