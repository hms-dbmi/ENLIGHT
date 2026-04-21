import torch.nn as nn


class NaiveRegress(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.num_class = conf.n_class
        self.dim_feat = conf.D_feat

        dropout  = 0.25
        size     = [self.dim_feat, 512, 256]
        self.phi = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        self.fc  = nn.Linear(size[2], self.num_class)

    def forward(self, feats):
        if feats.shape[0]==1:
            feats = feats.squeeze(0)
        x = self.phi(feats)
        x = self.rho(x)
        x = self.fc(x)
        return x
    
    def forward_to_loss(self, criterion, feats, labels):
        logits = self.forward(feats) #(K, 2)
        if labels.shape[0]==1:
            labels = labels.repeat(logits.shape[0])
        loss  = criterion(logits, labels)
        return loss, {}
    
    def infer_bag(self, x):
        x = self.forward(x)
        return x.mean(dim=0, keepdim=True)