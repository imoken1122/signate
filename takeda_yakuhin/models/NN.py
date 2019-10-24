import torch as th
from torch.autograd import Variable as V
from torch import nn,optim
from models.base_model import Model
class NN(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim//2),
           # nn.Dropout(0.6),   
            #nn.ELU(),
            nn.ReLU(True),
#            nn.LeakyReLU(True),
           nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim//2, input_dim//2),
            #nn.Dropout(0.5),       
           # nn.ELU(),
            nn.ReLU(True),
            #nn.LeakyReLU(True),   
          nn.BatchNorm1d(input_dim//2),
            nn.Linear(input_dim //2, 1))
        
        #nn.init.kaiming_normal_(self.layer.weight)
        #self.output = nn.Sigmoid()
        #self.output = nn.Softmax()
    def forward(self,x):
        x = self.layer(x)
        return x

class Nerualnet(Model):
    def train_and_predict(self,train,valid,test,param):
        INPUT

    def create_batch1(x,y,batch_size):
        a = list(range(len(x)))
        np.random.shuffle(a)
        x = x[a]
        y = y[a]

        batch_x = [x[batch_size * i : (i+1)*batch_size,:].tolist() for i in range(len(x)//batch_size)]
        batch_y = [y[batch_size * i : (i+1)*batch_size].tolist() for i in range(len(x)//batch_size)]
        return batch_x, batch_y