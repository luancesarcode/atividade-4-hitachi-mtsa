from torch import nn

class ConvRep(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.get("device", "cuda:0")
        self.dropout = cfg.get("dropout", 0.1)
        self.dilation = cfg.get("dilation", 1)
        self.output_datapoints = cfg.get("output_datapoints", 313)
        self.is_fitted = False
        self.history = {}
        feature_layers = eval(cfg["layers"])

        self.pool = nn.AdaptiveAvgPool1d(self.output_datapoints)
        in_d=1 # mono
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(feature_layers): 
            dim, kernel_size, stride, padding = cl
            is_last = (i == len(feature_layers) - 1)
            self.conv_layers.append(
                self.__block(in_d, dim, kernel_size, stride, padding, self.dropout, self.dilation, is_last=is_last)
            )
            in_d = dim
    
    def __block(self, c_in, c_out, kernel_size, stride, padding, dropout, dilation, is_last=False):
        layers = [
            nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        ]
        if not is_last:
            layers.extend([
                nn.BatchNorm1d(c_out),
                nn.Dropout(p=dropout),
                nn.GELU()
            ])
        return nn.Sequential(*layers)

    @property
    def name(self):
        return "CNN"
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        x = self.pool(x)
        return x