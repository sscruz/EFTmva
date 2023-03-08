import torch.nn as nn
import torch 

cost =  nn.BCELoss( reduction='mean')

class Net(nn.Module):
    def __init__(self, features=16, device='cpu'):
        super().__init__()
        self.main_module = nn.Sequential( 
            nn.Linear(features, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1 ),
            nn.Sigmoid(),
        )
        self.main_module.to(device)
    def forward(self, x):
        return self.main_module(x)
            

class Model:
    def __init__(self, features=16, device='cpu'):
        self.net = Net(features, device=device)


    def cost_from_batch(self, features, weight_sm, weight_bsm, device ): 

        index0 = torch.zeros( weight_sm.shape[0], 1, device=device) 
        index1 = torch. ones( weight_sm.shape[0], 1, device=device) 

        
        combined_label    = torch.cat( [index0, index1])
        combined_features = torch.cat( [features, features])
        combined_weight   = torch.cat( [weight_sm  / torch.mean(weight_sm), weight_bsm / torch.mean(weight_bsm)]) 

        combined_weight = torch.minimum( combined_weight, 1e3*torch.median(combined_weight)) # some regularization :) 

        cost.weight = combined_weight
        return cost( self.net(combined_features).flatten(), combined_label.flatten())
