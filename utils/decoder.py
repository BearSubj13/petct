import torch
from torch import nn
import torch.nn.functional as F


class Decoder1(nn.Module):
    def __init__(self, num_layers=5, num_features=128):
        super(Decoder1, self).__init__()
        self.fc1 = nn.Linear(num_features, num_features)
        self.ln1 = nn.LayerNorm(num_features)
        self.fc2 = nn.Linear(num_features, 256)
        self.ln2 = nn.LayerNorm(256)
        deconv_layers = []    
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1, stride=2, output_padding=1),
                                 nn.BatchNorm2d(8),  nn.LeakyReLU()))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, stride=2, output_padding=1),
                                 nn.BatchNorm2d(8),  nn.LeakyReLU()))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(8, 4, kernel_size=3, padding=1, stride=2, output_padding=1),
                                 nn.BatchNorm2d(4),  nn.LeakyReLU()))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(4, 4, kernel_size=3, padding=1, stride=2, output_padding=1),
                                 nn.BatchNorm2d(4),  nn.LeakyReLU()))
        deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(4, 1, kernel_size=3, padding=1, stride=2, output_padding=1) ))
                                 #nn.BatchNorm2d(2),  nn.LeakyReLU()))
        #deconv_layers.append(nn.ConvTranspose2d(2, 1, kernel_size=3, stride=1, padding=1, output_padding=0))

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.leaky_relu(x)
        #8*8*4 = 256
        x_reshaped = x.reshape(batch_size, 16, 4, 4)
        for deconv in self.deconv_layers:
            x_reshaped = deconv(x_reshaped)

        return x_reshaped


if __name__ == "__main__":
    model = Decoder1()
    embedding = torch.ones(128).unsqueeze(0)
    output = model(embedding)
    print(output.shape)