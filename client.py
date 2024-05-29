from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


# from grace:
import matplotlib.pyplot as plt
import numpy as np
import os

# Femnist specific
from FEMNIST_tests.femnist import FemnistDataset, FemnistNet, load_FEMNIST

# EuroSAT specific
from EUROSAT.data import EuroSATNet, load_EUROSAT

# #############################################################################
# Checking for Client Resources
# #############################################################################

# Use GPU on system if possible
warnings.filterwarnings("ignore", category=UserWarning)
if torch.cuda.is_available():
    print ("GPU CUDA")
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    print ("MPS device")
    DEVICE = torch.device("mps")
else:
    print ("MPS device not found, using CPU")
    DEVICE = torch.device("cpu")

# #############################################################################
# Training Loop
# #############################################################################

def train(net, trainloader, config, cid):
    """Train the model on the training set."""

    criterion_mean = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=float(config["learning_rate"]), momentum=float(config["momentum"]))
    net.train() 


    if 'duration' in config:
      total_epochs = int (float(config['duration'])-120) # / 60 / 5)
      if total_epochs > int(config['epochs']):
        total_epochs = int(config['epochs'])
      elif total_epochs < 0:
        total_epochs = 0
      if config['dataset'] == "EUROSAT":
        total_epochs = int(total_epochs/10)
    else:
      if 'model_type' in config:
        if config['model_type'] == "local_cluster":
          total_epochs = int(config['epochs'])
        else:
          total_epochs = -1
      else:
        total_epochs = int(config['epochs'])

    print(total_epochs)

    if total_epochs != -1:
      for epoch in range(total_epochs):
          print("Epoch: "+str(epoch))
          
          for images, labels in trainloader:
            images = images.to(DEVICE)      # @ make sure to set images/labels to the device you're using
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            if 'duration' in config:
              global_params = [val.detach().clone() for val in net.parameters()]
              proximal_mu = float(config["prox_term"])
              proximal_term = 0
              for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += torch.square((local_weights - global_weights).norm(2))
              loss = criterion_mean(net(images), labels) + (proximal_mu / 2) * proximal_term
            else:
              loss = criterion_mean(net(images), labels)
            loss.backward()
            optimizer.step()
          

# #############################################################################
# Validation Loop
# #############################################################################

def test(net, testloader):
  """Validate the model on the test set."""
  net.eval()

  criterion = torch.nn.CrossEntropyLoss()
  correct, total, loss = 0, 0, 0.0
  with torch.no_grad():
    for images, labels in testloader:
      outputs = net(images.to(DEVICE))
      loss += criterion(outputs, labels.to(DEVICE)).item()
      total += labels.size(0)
      # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
      correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item() # add .toDevice to labels if using GPU
  return loss / len(testloader.dataset), correct / total


# #############################################################################
# Federating the pipeline with Flower
# #############################################################################


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def __init__(self,
                 cid: int,
                 net: nn.Module,
                 trainloader: DataLoader,
                 testloader: DataLoader):
        
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(self.net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.net.load_state_dict(state_dict, strict=True)

  def save_local_model(self, config):
    name = config['name']
    alg = config['alg']

    folder_name = f'/datasets/{alg}/model_files_{name}/'
    print(folder_name)

    if not os.path.exists(folder_name):
      os.makedirs(folder_name)
    
    if alg == "AutoFLSat2":
      if config['model_update'] == 'global_cluster':
        for cluster in range(1,int(config['n_cluster'])+1):
          file_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{cluster}.pth'
          torch.save(self.net.state_dict(), file_name)
      else:
        cluster = config['cluster_identifier']
        agg_cluster = config['agg_cluster']
        file_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{agg_cluster}.pth'
        torch.save(self.net.state_dict(), file_name)

    else:
      if alg == "AutoFLSat":
        cluster = config['cluster_identifier']
        agg_cluster = config['agg_cluster']
        file_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{agg_cluster}.pth'
      else:
        file_name = f'/datasets/{alg}/model_files_{name}/{self.cid}.pth'
      torch.save(self.net.state_dict(), file_name)

  def fit(self, parameters, config):
    name = config['name']
    alg = config['alg']
    buff_name = ""
    auto_name = ""

    if alg == "AutoFLSat":
      cluster = config['cluster_identifier']
      agg_cluster = config['agg_cluster']
      auto_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{agg_cluster}.pth'
    else:
      buff_name = f'/datasets/{alg}/model_files_{name}/{self.cid}.pth'
    # check if there is a local model saved to the disk, if so use that (FedBuff)

    if os.path.exists(buff_name) and config['alg'].startswith("FedBuff"):
      self.net.load_state_dict(torch.load(buff_name))
    elif os.path.exists(auto_name) and config['alg'].startswith("AutoFLSat"):
      self.net.load_state_dict(torch.load(auto_name))
    else:
      self.set_parameters(parameters)

    train(self.net, self.trainloader, config, self.cid)

    return self.get_parameters(config={}), len(self.trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    if 'model_update' in config:
      self.save_local_model(config)
      print(self.cid)
      print("work pls: "+str(config['model_update']))

      # delete mid models of other clusters after agg
      if config['model_update'] == 'global_cluster' and config['alg'] == "AutoFLSat":
        name = config['name']
        alg = config['alg']
        cluster_n = config['n_cluster']
        cluster = config['cluster_identifier']

        for agg_cluster in range(1,int(cluster_n)+1):
            if int(cluster) != int(agg_cluster):
              file_name = f'/datasets/{alg}/model_files_{name}/{cluster}_{str(agg_cluster)}.pth'
              if os.path.exists(file_name):
                os.remove(file_name)
                print(file_name)
                print("deleted")

    loss, accuracy = test(self.net, self.testloader)
    return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn_femnist(cid: int) -> FlowerClient:

    # Load model and data
    print("MADE CLIENT")
    if torch.cuda.is_available():
        print ("GPU CUDA")
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print ("MPS device")
        DEVICE = torch.device("mps")
    else:
        print ("MPS device not found, using CPU")
        DEVICE = torch.device("cpu")
        
    net = FemnistNet().to(DEVICE)
    trainloader, testloader = load_FEMNIST(cid)
    
    return FlowerClient(cid, net, trainloader, testloader).to_client()

def client_fn_EuroSAT(cid: int) -> FlowerClient:

    # Load model and data
    print("MADE CLIENT")
    if torch.cuda.is_available():
        print ("GPU CUDA")
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print ("MPS device")
        DEVICE = torch.device("mps")
    else:
        print ("MPS device not found, using CPU")
        DEVICE = torch.device("cpu")
        



    net = EuroSATNet().to(DEVICE)
    trainloader, testloader = load_EUROSAT(cid)
    
    return FlowerClient(cid, net, trainloader, testloader).to_client()