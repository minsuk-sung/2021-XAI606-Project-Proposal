import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm

def get_network(
    options
):
    model = None

    if options.network == "ResNet":        
        model = models.resnet18(pretrained=options.model.pretrained)
        model.fc = nn.Linear(512, options.data.num_classes)

    else:
        raise NotImplementedError

    return model.to(options.device)

def get_optimizer(
    params,
    options
):
    if options.optimizer.type == "Adam":
        optimizer = optim.Adam(params, lr=options.optimizer.lr, weight_decay=options.optimizer.weight_decay)
    else:
        raise NotImplementedError

    return optimizer
