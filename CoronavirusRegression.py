import torch
import matplotlib.pyplot as plt
import csv
 
def csv_dict_reader(file_obj): # just read data from datanew.csv
    reader = csv.DictReader(file_obj, delimiter=',')
    sp = []
    for line in reader:
        if line["countriesAndTerritories"] == "Russia" and int(line["month"]) >= 2:
            sp.append(int(line["cases"]))
    return sp[:] # returns list of cases of coronavirus

with open("datanew.csv") as f_obj:
    x_new = csv_dict_reader(f_obj) # get the list for training

class RegressionNet(torch.nn.Module): # the structure of NN
    def __init__(self, n_hidden):
        super(RegressionNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidden)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden, 1)
       
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x
    
def predict(net, x, y): # How to built the diagramm
    y_predicted = net.forward(x)
    
    plt.plot(x.data, y_predicted.data, color="blue")
    plt.plot(x.data, y, 'o', c="orange")
    plt.show()
    

net = RegressionNet(50000)

x_train = torch.tensor(list(range(1, len(x_new)+1))).float() # prepare the data
y_train =  torch.tensor(x_new[::-1]).float()

x_train.unsqueeze_(1) # converting from list to vector
y_train.unsqueeze_(1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.01) 

def loss(pred, target): # loss-function is MSE
    return ((pred - target)**2).mean()

for epoch_index in range(2000): # the gradient descent
    optimizer.zero_grad() # make the grad equals 0, because pytorch is summing the grads

    y_pred = net.forward(x_train) # get the predicted values (this is matrix)
    loss_value = loss(y_pred, y_train) # calculation the loss-value
    loss_value.backward() # calculationg the grad
    optimizer.step() # make the step of gradient descent

predict(net, x_train, y_train) # show the diagramm

k = 7 # amount of prediction
x_future = torch.tensor(list(range(len(x_new)+1, len(x_new)+1 + k))).float().unsqueeze_(1) # predict the future values
print(net.forward(x_future)) # get a list of predicted values