import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm

# --- 1. Basic Modules ---

class RevIN(nn.Module):
    """ Reversible Instance Normalization """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x

class moving_avg(nn.Module):
    """ Moving average block """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """ Series decomposition block """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class TCNBlock(nn.Module):
    """ A residual block for Temporal Convolutional Network """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.1):
        super(TCNBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding='same', dilation=dilation))
        self.net = nn.Sequential(self.conv1, nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    """ Temporal Convolutional Network predictor """
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.out_proj = nn.Linear(input_dim * num_channels[-1], output_dim)
    
    def forward(self, x):
        # x shape: (B*C, L_in)
        x = x.unsqueeze(1)  # -> (B*C, 1, L_in)
        x = self.network(x) # -> (B*C, C_out, L_in)
        x = x.flatten(1)    # -> (B*C, C_out * L_in)
        return self.out_proj(x)

class MLP_Block(nn.Module):
    """ A simple MLP block """
    def __init__(self, input_dim, output_dim, hidden_dim_ratio=4, dropout_p=0.2):
        super(MLP_Block, self).__init__()
        hidden_dim = int(input_dim * hidden_dim_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.mlp(x)

# --- 2. Core Model Modules ---

class MultiScaleDecompBlock(nn.Module):
    """ Encoder block """
    def __init__(self, kernel_size):
        super(MultiScaleDecompBlock, self).__init__()
        self.downsample = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.decomp = series_decomp(kernel_size)

    def forward(self, x):
        high_freq, low_freq = self.decomp(x)
        low_freq_downsampled = self.downsample(low_freq.permute(0, 2, 1)).permute(0, 2, 1)
        return low_freq_downsampled, high_freq

class LatentTrendPredictor(nn.Module):
    """ Bottleneck predictor """
    def __init__(self, input_len, pred_len, channels):
        super(LatentTrendPredictor, self).__init__()
        self.predictor = MLP_Block(input_len, pred_len)
    
    def forward(self, x):
        B, L, C = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * C, L)
        y_flat = self.predictor(x_flat)
        y = y_flat.reshape(B, C, -1).permute(0, 2, 1)
        return y

class MultiScaleFusionBlock(nn.Module):
    """ Decoder block """
    def __init__(self, channels, skip_len, out_len, tcn_channels, use_gate=True):
        super(MultiScaleFusionBlock, self).__init__()
        self.use_gate = use_gate
        self.detail_predictor = TCN(skip_len, out_len, num_channels=tcn_channels)
        self.upsample = nn.Upsample(size=out_len, mode='linear', align_corners=True)
        
        if self.use_gate:
            self.gate = nn.Sequential(nn.Linear(out_len, out_len), nn.Sigmoid())

    def forward(self, x_deep_pred, x_high_skip):
        B, _, C = x_deep_pred.shape
        y_up = self.upsample(x_deep_pred.permute(0, 2, 1)).permute(0, 2, 1)
        
        L_skip = x_high_skip.shape[1]
        x_high_flat = x_high_skip.permute(0, 2, 1).contiguous().view(B * C, L_skip)
        
        y_high_pred_flat = self.detail_predictor(x_high_flat)
        y_high_pred = y_high_pred_flat.view(B, C, -1).permute(0, 2, 1)
        
        if self.use_gate:
            gate_weights = self.gate(y_up.permute(0, 2, 1)).permute(0, 2, 1)
            y_out = y_up + gate_weights * y_high_pred
        else:
            y_out = y_up + y_high_pred
            
        return y_out

# --- 3. Main Model ---

class Model(nn.Module):
    """
    The original baseline HDN/U-Net model.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        # Basic configs from the provided Namespace object
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        print(self.seq_len, self.pred_len, self.channels)
        # --- Key Hyperparameters for this baseline model ---
        # You can expose these in `configs` for easier tuning
        self.num_levels = 3
        # Use dynamic kernel sizes as it's a proven improvement
        base_kernel_size = getattr(configs, 'moving_avg', 25) 
        self.kernel_sizes = [max(3, (25 // (2**i)) // 2 * 2 + 1) for i in range(self.num_levels)]
        print(self.kernel_sizes)

        self.tcn_channels = [16, 32] # Default TCN channels
        self.use_gate = True

        self.revin_layer = RevIN(self.channels, affine=True)

        # Encoder setup
        self.encoder = nn.ModuleList([
            MultiScaleDecompBlock(kernel_size=k) for k in self.kernel_sizes
        ])

        # Bottleneck predictor setup
        latenttrend_in_len = max(1, self.seq_len // (2**self.num_levels))
        latenttrend_out_len = max(1, self.pred_len // (2**self.num_levels))
        self.latenttrend = LatentTrendPredictor(latenttrend_in_len, latenttrend_out_len, self.channels)

        # Decoder setup
        self.decoder = nn.ModuleList()
        for i in range(self.num_levels):
            level = self.num_levels - 1 - i
            skip_len = max(1, self.seq_len // (2**level))
            out_len = max(1, self.pred_len // (2**level))
            self.decoder.append(MultiScaleFusionBlock(
                self.channels, 
                skip_len, 
                out_len, 
                tcn_channels=self.tcn_channels,
                use_gate=self.use_gate
            ))

    def forward(self, x):
        x_norm = self.revin_layer(x, 'norm')
        
        skip_connections = []
        x_current = x_norm
        for i in range(self.num_levels):
            if x_current.size(1) < 2: break
            x_down, x_high = self.encoder[i](x_current)
            skip_connections.append(x_high)
            x_current = x_down
        
        y_latenttrend = self.latenttrend(x_current)
        
        y_current = y_latenttrend
        for i in range(self.num_levels):
            x_high_skip = skip_connections[self.num_levels - 1 - i]
            y_current = self.decoder[i](y_current, x_high_skip)
            
        if y_current.size(1) != self.pred_len:
            y_current = F.interpolate(y_current.permute(0,2,1), size=self.pred_len, mode='linear', align_corners=True).permute(0,2,1)
            
        final_prediction = self.revin_layer(y_current, 'denorm')
        return final_prediction