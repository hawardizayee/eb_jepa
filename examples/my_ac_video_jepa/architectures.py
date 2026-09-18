from typing import Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from nn_utils import init_module_weights

class Projector(nn.Module):
    """MLP projector built from a spec string line '256-512-128'."""

    def __init__(self, mlp_spec):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1] # Store output dimension as attribute

    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    """ResNet Block."""

    def __init__(self, num_features):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)

class ResnetStack(nn.Module):
    """ResNet stack module."""
    
    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super(ResnetStack, self).__init__()
        self.num_features = num_features
        self.num_blocks = num_blocks
        self.max_pooling = max_pooling
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        if max_pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.max_pool = nn.Identity()

    def forward(self,x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x 

class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    def __init__(
        self,
        width=1,
        stack_sizes=(16,32,32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=2,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(2,65,65)
    ):
        super(ImpalaEncoder, self).__init__()
        self.width = width 
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        input_channels = [input_channels] + list(stack_sizes)   # [2,16,32,32]

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=input_channels[i],   # [2, 16, 32, 32]
                    num_features=stack_size * width,    # (16, 32, 32)
                    num_blocks=num_blocks,              # 2 ; ResnetBlock 
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        # Compute MLP input dimension dynamically 
        with torch.no_grad():
            # Create a dummy input (assuming typical input size for this encoder)
            dummy_input = torch.zeros(1, *self.input_shape) # (1, C=2, H=65, W=65)
            conv_out = dummy_input
            for stack_block in self.stack_blocks:
                conv_out = stack_block(conv_out)    # b c w h 
            flattened_dim = conv_out.view(conv_out.size(0), -1).shape[1] # c * w * h = 2592

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)

        if final_ln:   # True
            self.final_ln = nn.LayerNorm(self.mlp_output_dim)
        else:
            self.final_ln = nn.Identity()

    
    def forward(self,x):
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            out: [B, D, T, 1, 1]
        """

        # [B, C, T, H, W] --> [T, B, C, H, W]
        (
            _, 
            _, 
            t,
            _,
            _,
        ) = x.shape
        x = x.permute(2, 0, 1, 3, 4)

        features = []

        for i in range(t):

            conv_out = x[i]

            for i, stack_block in enumerate(self.stack_blocks):
                conv_out = stack_block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)                

            conv_out = F.relu(conv_out)
            if self.layer_norm: # False 
                conv_out = nn.LayerNorm(conv_out.size()[1:])(conv_out) # b c w h
            #flatten 
            out = conv_out.view(conv_out.size(0), -1) # (B, c * w * h = 2592)
            out = self.mlp(out)
            out = self.final_ln(out)

            features.append(out)

        features = torch.stack(features, dim=1)

        features = features.transpose(1,2).unsqueeze(-1).unsqueeze(-1)

        return features
    
class RNNPredictor(nn.Module):
    """GRU-based predictor for single-step state propagation."""

    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: Optional[int] =2,
        num_layers: int = 1,
        final_ln: Optional[torch.nn.Module] = None, 
    ):
        super(RNNPredictor, self).__init__()

        self.num_layers = num_layers

        self.rnn = torch.nn.GRU(
            input_size=action_dim,      # 2
            hidden_size=hidden_size,    # 512
            num_layers=num_layers       # 1 
        )

        self.final_ln = final_ln
        self.is_rnn = True
        self.context_length = 0 

    def forward(self, state, action): # context_states, context_actions
        """
        Propagate one step forward.

        Args:
            state: [B, D, 1, 1, 1]
            action: [B, A, 1]
        Returns:
            next_state: [B, D, 1, 1, 1]
        """
        # This only does one step
        rnn_state = state.flatten(1, 4).unsqueeze(0).contiguous() # [1, B, D]
        rnn_input = action.squeeze(-1).unsqueeze(0).contiguous() # [1, B, A]

        next_state, _ = self.rnn(rnn_input, rnn_state)      # (1, B=32, hidden_size = 512)
        
        next_state = self.final_ln(next_state)

        return next_state[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    

class InverseDynamicsModel(nn.Module):
    """
    Predicts the action that caused a transition from state_t to state_t_plus_t.
    Used as auxiliray task for representation learning. 
    """

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_t_plus_1):
        """
        Args:
            state_t: State at time t, shape [B, D]
            state_t_plus_1: State at time t+1, shape [B, D]
        Returns:
            predicted_action: Action predicted to transform state_t to state_t_plus_1, shape [B, A]
        """
        combined_states = torch.cat([state_t, state_t_plus_1], dim=1)
        return self.model(combined_states)