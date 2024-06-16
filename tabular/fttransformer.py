class FT(nn.Module):
    def __init__(self, 
                 num_heads, 
                 dim_feedforward, 
                 dropout, 
                 in_features, 
                 embed_dim, 
                 num_layers, 
                 num_class):
        super(FT, self).__init__()
        assert num_heads % embed_dim == 0, "num_head mist be divisible by embed_dim"

        self.embedding = nn.Linear(in_features, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 
                                       num_heads, 
                                       dim_feedforward, 
                                       dropout, 
                                       batch_first=True),
            num_layers
        )
        self.out = nn.Linear(embed_dim, num_class)


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return F.relu(self.out(x))