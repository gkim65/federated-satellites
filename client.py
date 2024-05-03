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


# Femnist specific
from FEMNIST_tests.femnist import FemnistDataset, FemnistNet, load_FEMNIST


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
      total_epochs = int (float(config['duration']) / 60 / 5)
    else:
      total_epochs = int(config['epochs'])

    print(total_epochs)

    for epoch in range(total_epochs):
        print("Epoch: "+str(epoch))
        
        for images, labels in trainloader:
          images = images.to(DEVICE)      # @ make sure to set images/labels to the device you're using
          labels = labels.to(DEVICE)
          optimizer.zero_grad()

          if config["alg"] == "fedProxSat" or config["alg"] == "fedProx2Sat":
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
            
            ### FROM HERE:
            # # Train the local model w/ our data, biased by the difference from the global model
            # local_optimizer.zero_grad()

            # # Update local model
            # local_loss = criterion_mean(local_net(images), labels)
            # local_loss.backward()
            # for local_param, global_param in zip(local_net.parameters(), init_global_net.parameters()):
            #     local_param.grad += config['lambda'] * (local_param - global_param)
            # local_optimizer.step()

            # # Update global model
            # global_optimizer.zero_grad()
            # global_loss = criterion_mean(global_net(images), labels)
            # global_loss.backward()
            # global_optimizer.step()
    
    # save_data(losses, images_passed, images_failed, config['test_name'], cid, DEVICE)


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

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(self.net, self.trainloader, config, self.cid)
    return self.get_parameters(config={}), len(self.trainloader.dataset), {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(self.net, self.testloader)
    return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn(cid: int) -> FlowerClient:

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
        
    # TODO: need to find a way to fix this, how to switch specific datasets
    # if config_object["TEST_CONFIG"]["dataset"] == "FEMNIST":
    net = FemnistNet().to(DEVICE)
    trainloader, testloader = load_FEMNIST(cid)
    
    return FlowerClient(cid, net, trainloader, testloader).to_client()