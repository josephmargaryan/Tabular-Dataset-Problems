class BaseLine(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(BaseLine, self).__init__()
        
        self.fcl1 = nn.Linear(in_channels, hidden_dim)
        self.fcl2 = nn.Linear(hidden_dim, hidden_dim)
        self.fcl3 = nn.Linear(hidden_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = F.relu(self.fcl1(x))
        x = F.relu(self.fcl2(x))
        return self.fcl3(x)