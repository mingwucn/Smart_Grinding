import torch
import torchaudio
import torchvision
import torchvision.transforms.v2
import torch.nn as nn
import torchaudio.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import sklearn


class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNetEncoder(nn.Module):
    def __init__(self, block, num_blocks, input_shape, latent_dim):
        super(ResNetEncoder, self).__init__()
        self.in_planes = 64
        self.input_shape = input_shape

        input_channels = input_shape[0]
        
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, latent_dim)
        
        # Automatically infer the size before fully connected layer
        # self.flatten_size = self._get_flatten_size()
        # self.out_size = self._get_out_size()
        # self.flatten_size = self.out_size.numel()
        
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    # def _get_flatten_size(self):
    #     # Create a dummy input to infer the size
    #     dummy_input = torch.zeros(1, *self.input_shape)
    #     out = self.forward_features(dummy_input)
    #     return out.view(out.size(0), -1).size(1)

    # def _get_out_size(self):
    #     # Create a dummy input to infer the size
    #     dummy_input = torch.zeros(1, *self.input_shape)
    #     out = self.forward_features(dummy_input)
    #     return out.size()[1:]

    def forward_features(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1)
        # out = torch.nn.functional.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        return x

    def forward(self, x):
        out = self.forward_features(x)
        out = torch.nn.Flatten()(out)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

class ResNetDecoder(nn.Module):
    def __init__(self, BasicBlockDec, num_blocks, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_planes = 512
        self.output_shape = output_shape
        # self.out_size = out_size

        self.fc = nn.Linear(latent_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, output_shape[0], kernel_size=3, scale_factor=3)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 512, 1, 1)
        # out = out.view(out.size(0), 512, 8, 8)
        # out = out.view(out.size(0),*list(self.out_size))

        x = torch.nn.functional.interpolate(x, scale_factor=16)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        # print(x.shape)
        x = torch.nn.functional.interpolate(x, self.output_shape[1:])
        return x

class VAE(nn.Module):
    def __init__(self, blockEC, blockDC, num_blocks, input_shape, latent_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.encoder = ResNetEncoder(blockEC, num_blocks, input_shape, latent_dim)
        # print("Output size:", self.encoder.out_size)
        self.decoder = ResNetDecoder(blockDC, num_blocks, latent_dim, input_shape)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def ResNetVAE(input_shape, latent_dim):
    return VAE(BasicBlockEnc, BasicBlockDec, [2, 2, 2, 2], input_shape, latent_dim)

class SVMModel(nn.Module):
    def __init__(self, input_dim, kernel='rbf', num_classes=2):
        super().__init__()
        self.kernel_type = kernel
        self.svm = nn.Linear(input_dim, num_classes)


    def forward(self, time_series_data, metadata:list = None):
        # x = torch.cat([time_series_data,*metadata],dim=1)
        x = torch.cat([time_series_data,*[i.reshape(-1,1) for i in metadata]],dim=1)

        if self.kernel_type == 'rbf':
            x = self.rbf_kernel(x)
        elif self.kernel_type == 'poly':
            x = self.poly_kernel(x)
        elif self.kernel_type == 'sigmoid':
            x = self.sigmoid_kernel(x)
        logits = self.svm(x)
        return logits
    
    def rbf_kernel(self, x, gamma=0.1):
        """ Radial Basis Function (Gaussian) Kernel. """
        # Compute pairwise squared distances
        pairwise_sq_dists = torch.norm(x[:, None] - x, dim=1).pow(2)
        return torch.exp(-gamma * pairwise_sq_dists)
    
    def poly_kernel(self, x, degree=3):
        """ Polynomial Kernel. """
        # return (torch.matmul(x, x.T) + 1) ** degree
        return (x* x + 1) ** degree
    
    def sigmoid_kernel(self, x, alpha=0.1, c=0):
        """ Sigmoid Kernel. """
        # return torch.tanh(alpha * torch.matmul(x, x.T) + c)
        return torch.tanh(alpha * (x* x) + c)

class CNN_Base_1D_Model(nn.Module):
    def __init__(self,time_series_length=1920, meta_data_size=4, meta_network_out_ratio = 4, num_classes=3,dropout=0.2):
        super().__init__()
        self.time_series_length = time_series_length
        self.meta_data_size = meta_data_size
        meta_network_out = meta_data_size*meta_network_out_ratio
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(self._get_conv_output_size()+meta_network_out, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, num_classes)  
        if self.meta_data_size > 0:
            self.meta_fc1 = nn.Linear(meta_data_size, 32)  
            self.meta_fc2 = nn.Linear(32, 64)  
            self.meta_fc3 = nn.Linear(64, meta_network_out) 

    def _get_conv_output_size(self):
        # Create a dummy input with the same dimensions as the actual input
        dummy_input = torch.zeros(1, 1, self.time_series_length)  # Batch size, channels, input length
        output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(output.size()[1:])))  # Flattened size

    def _forward_conv(self, x):
        # x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))

        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, time_series_data, metadata:list = None):
        x = time_series_data.unsqueeze(1)  # Add a channel dimension
        x = self._forward_conv(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers

        if self.meta_data_size>0:
            # Meta network
            metadata = torch.stack(metadata, dim=1)
            metadata = self.meta_fc1(metadata)
            metadata = self.meta_fc2(metadata)
            metadata = self.meta_fc3(metadata)
            
            # Concatenate metadata with the CNN output
            x = torch.cat([x, metadata], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))  # Binary outputs
        return output

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = nn.functional.relu(out)
        return out

class ResNet15_1D_Model(nn.Module):
    def __init__(self, block = ResidualBlock1D, layers=[2, 2, 2, 2], num_classes=2,time_series_length=1920, meta_data_size=4, meta_network_out_ratio=4,dropout=0.2):
        super().__init__()
        self.time_series_length = time_series_length
        self.in_channels = 64
        meta_network_out = meta_data_size*meta_network_out_ratio
        self.meta_data_size = meta_data_size
        # Initial convolution and batch normalization layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(dropout)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Meta network
        if self.meta_data_size > 0:
            self.meta_fc1 = nn.Linear(meta_data_size, 32)  
            self.meta_fc2 = nn.Linear(32, 64)  
            self.meta_fc3 = nn.Linear(64, meta_network_out) 
        
        # Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self._get_conv_output_size()+meta_network_out, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        # print(block)
        # print(block(self.in_channels, out_channels, stride, downsample))
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
   
    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, self.time_series_length)  # Batch size, channels, input length
        output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(output.size()[1:])))

    def _forward_conv(self,x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return x
    
    def forward(self, time_series_data, metadata:list = None):
        x = time_series_data.unsqueeze(1)  # Input shape [batch_size, 1, 1920]
        x = self._forward_conv(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)

        if self.meta_data_size>0:
            # Meta network
            metadata = torch.stack(metadata, dim=1)
            metadata = self.meta_fc1(metadata)
            metadata = self.meta_fc2(metadata)
            metadata = self.meta_fc3(metadata)
            # Concatenate metadata with the CNN output
            x = torch.cat([x, metadata], dim=1)
        
        x = self.fc(x)
        return torch.sigmoid(x)

def make_4d_resnet(input_c,classes):
    import pytorchvideo.models.resnet
    """(C, T, H, W)"""
    return pytorchvideo.models.resnet.create_resnet(
        input_channel=input_c,  # RGB input from Kinetics
        model_depth=50,   
        model_num_class=classes,  # Kinetics has 3 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )

def make_x3d(input_c,classes):
    import pytorchvideo.models.x3d
    return pytorchvideo.models.x3d.create_x3d(
        input_channel=input_c,  # RGB input from Kinetics
        model_num_class=classes,  # Kinetics has 3 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU, 
    )

# 1D Hidden Markov Model (HMM)
class HMM_1D(nn.Module):
    def __init__(self, sequences, labels, n_components=5):
        super().__init__()
        self.n_components = n_components
        self.model = hmm.GaussianHMM(n_components=n_components)
        self.sequences = sequences
        self.labels = labels

    def train_hmm(self):
        model = self.model
        X = np.array(self.sequences)
        model.fit(X)
        return model

    def predict_hmm(self):
        logprob, _ = self.model.decode(self.sequence)
        return logprob

# 1D Convolutional Neural Network (CNN)
class CNN_1D(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4, 128)  # Adjust based on output of conv layers
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x.unsqueeze(1))))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4)  # Adjust based on output of conv layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 1D LSTM
class LSTM_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to class predictions
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, metadata=None):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Pass the input through the LSTM
        out, _ = self.lstm(x.unsqueeze(2), (h0, c0))
        
        # Use the last hidden state for classification
        out = self.fc(out[:, -1, :])
        return out

class RNN_1D(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, metadata=None):
        embedded = self.embedding(x)  # Shape: (batch_size, seq_len, embedding_dim)
        output, hidden = self.rnn(embedded)  # Output shape: (batch_size, seq_len, hidden_dim)
        hidden = hidden.squeeze(0)  # Take the last hidden state, Shape: (batch_size, hidden_dim)
        return self.fc(hidden)  # Shape: (batch_size, output_dim)

# 1D Transformer (Simplified)
class Transformer_1D(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, metadata=None):
        output = self.transformer_encoder(src)
        return output[:, 0, :]  # Use the first output token

# 1D Conditional Random Fields (CRF) (Using pytorch-crf)
# (Requires installation: pip install torchcrf)

class CRF_1D(nn.Module):
    def __init__(self, input_size, tagset_size, hidden_dim):
        super().__init__()
        from TorchCRF import CRF
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=1, bidirectional=True)
        self.crf = CRF(tagset_size)

    def forward(self, emissions, tags, mask=None):
        # emissions: (seq_len, batch_size, 2*hidden_dim)
        # tags: (seq_len, batch_size)
        # mask: (seq_len, batch_size)

        lstm_out, _ = self.lstm(emissions)
        lstm_out = lstm_out.view(emissions.size(0), emissions.size(1), self.hidden_dim * 2)

        # Compute the negative log-likelihood
        loss = -self.crf(emissions, tags, mask=mask) 
        return loss

    def decode(self, emissions, mask=None):
        return self.crf.decode(emissions, mask=mask)

class CNN_Base_2D_Model(nn.Module):
    def __init__(self,input_shape, meta_data_size=4, meta_network_out_ratio = 4, num_classes=3,dropout=0.2):
        super().__init__()
        self.input_shape = input_shape
        self.meta_data_size = meta_data_size
        meta_network_out = meta_data_size*meta_network_out_ratio
        # self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(self._get_conv_output_size()+meta_network_out, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, num_classes)  
        if self.meta_data_size > 0:
            self.meta_fc1 = nn.Linear(meta_data_size, 32)  
            self.meta_fc2 = nn.Linear(32, 64)  
            self.meta_fc3 = nn.Linear(64, meta_network_out) 

    def _get_conv_output_size(self):
        # Create a dummy input with the same dimensions as the actual input
        dummy_input = torch.zeros(1, 1, self.input_shape[0],self.input_shape[1])  # Batch size, channels, input length
        output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(output.size()[1:])))  # Flattened size

    def _forward_conv(self, x):
        # x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        # x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))

        x = self.pool1(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool2(nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool3(nn.functional.relu(self.bn3(self.conv3(x))))
        return x

    def forward(self, time_series_data, metadata:list = None):
        x = time_series_data.unsqueeze(1)  # Add a channel dimension
        x = self._forward_conv(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers

        if self.meta_data_size>0:
            # Meta network
            metadata = torch.stack(metadata, dim=1)
            metadata = self.meta_fc1(metadata)
            metadata = self.meta_fc2(metadata)
            metadata = self.meta_fc3(metadata)
            
            # Concatenate metadata with the CNN output
            x = torch.cat([x, metadata], dim=1)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))  # Binary outputs
        return output

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = nn.functional.relu(out)
        return out

class ResNet15_2D_Model(nn.Module):
    def __init__(self, block = ResidualBlock2D, layers=[2, 2, 2, 2], num_classes=2,input_shape=(20,188), meta_data_size=4, meta_network_out_ratio=4,dropout=0.2):
        super().__init__()
        self.input_shape = input_shape
        self.in_channels = 64
        meta_network_out = meta_data_size*meta_network_out_ratio
        self.meta_data_size = meta_data_size
        # Initial convolution and batch normalization layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.drop = nn.Dropout(dropout)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Meta network
        if self.meta_data_size > 0:
            self.meta_fc1 = nn.Linear(meta_data_size, 32)  
            self.meta_fc2 = nn.Linear(32, 64)  
            self.meta_fc3 = nn.Linear(64, meta_network_out) 
        
        # Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self._get_conv_output_size()+meta_network_out, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = []
        # print(block)
        # print(block(self.in_channels, out_channels, stride, downsample))
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def _get_conv_output_size(self):
        dummy_input = torch.zeros(1, 1, *self.input_shape)  # Batch size, channels, height, width
        output = self._forward_conv(dummy_input)
        return int(torch.prod(torch.tensor(output.size()[1:])))

    def _forward_conv(self,x):
        x = self.conv1(x) 
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        return x
    
    def forward(self, time_series_data, metadata:list = None):
        x = time_series_data.unsqueeze(1)  # Input shape [batch_size, 1, 1920]
        x = self._forward_conv(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)

        if self.meta_data_size>0:
            # Meta network
            metadata = torch.stack(metadata, dim=1)
            metadata = self.meta_fc1(metadata)
            metadata = self.meta_fc2(metadata)
            metadata = self.meta_fc3(metadata)
            # Concatenate metadata with the CNN output
            x = torch.cat([x, metadata], dim=1)
        
        x = self.fc(x)
        return torch.sigmoid(x)


class LinearRegression_1D(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return self.linear(x)

    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.1)
            nn.init.constant_(m.bias, 0.5)

    def initialize_weights(self):
        """Apply weight initialization to all submodules."""
        self.apply(self._init_weights)

class MLP_1D(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64,32], output_dim=1, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_features = input_dim
        
        # Construct hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        
        # Add output layer
        layers.append(nn.Linear(in_features, output_dim))
        
        # Combine all layers into a sequential container
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _init_weights(self, module):
        """Initialize weights for Linear layers using Kaiming initialization."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def initialize_weights(self):
        """Apply weight initialization to all submodules."""
        self.apply(self._init_weights)


class ConvProcessor_2D(nn.Module):
    def __init__(self, input_size, conv_channels=[1,16,32,64], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_size=10):
        super().__init__()
        # Ensure kernel_size, stride, padding are tuples for flexibility
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        layers = []
        for i in range(len(conv_channels) - 1):
            layers.append(nn.Conv2d(conv_channels[i], conv_channels[i+1],
                                    kernel_size=self.kernel_size,
                                    stride=self.stride,
                                    padding=self.padding))
            layers.append(nn.BatchNorm2d(num_features=conv_channels[i+1])) 
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))) # MaxPool also often benefits from tuple for kernel/stride
        self.conv_sequence = nn.Sequential(*layers)
        conv_output_size = self._get_output_size(input_shape=input_size)
        self.fc = nn.Linear(conv_output_size, output_size)
        
    def forward(self, x):
        x = self.conv_sequence(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _get_output_size(self, input_shape):
        """
        Calculates the flattened output size of the convolutional layers
        given the input shape and conv channel config.
        input_shape: (C, H, W) of the input tensor.
        conv_channels: Configuration for the convolutional layers.
        """
        dummy_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            # Pass the dummy input through the conv_sequence
            conv_output = self.conv_sequence(dummy_input)
        return conv_output.view(1, -1).size(1)


class FcProcessor_2D(nn.Module):
    def __init__(self, neuron_config):
        super().__init__()
        layers = []
        for i in range(len(neuron_config) - 1):
            layers.append(nn.Linear(neuron_config[i], neuron_config[i+1]))
            if i < len(neuron_config) - 2:
                layers.append(nn.ReLU())
        self.fc_sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc_sequence(x)


class Fusion_2D(nn.Module):
    def __init__(self, input_feature_sizes, output_size):
        super().__init__()
        total_input_size = sum(input_feature_sizes)
        self.output_layer = nn.Linear(total_input_size, output_size)

    def forward(self, branch_outputs):
        if not branch_outputs:
            raise ValueError("No branch outputs provided for fusion.")
        combined_output = torch.cat(branch_outputs, dim=1)
        final_prediction = self.output_layer(combined_output)
        return final_prediction

