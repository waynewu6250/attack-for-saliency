from torch import nn

# Generator Model
class Generator(nn.Module):
    def __init__(self, inf, gnf):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
                    # Input dimension: inf x 1 x 1
                    nn.ConvTranspose2d(inf, gnf*32, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(gnf*32),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf*32, gnf*16, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(gnf*16),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf*16, gnf*8, 4, 1, 1, bias=False),
                    nn.BatchNorm2d(gnf*8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf*8, gnf*4, 4, 3, 1, bias=False),
                    nn.BatchNorm2d(gnf*4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf*4, gnf*2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(gnf*2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf*2, gnf, 8, 2, 1, bias=False),
                    nn.BatchNorm2d(gnf),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(gnf, 3, 12, 2, 1, bias=False),
                    nn.Tanh()
                    # Output dimension: 3 x 224 x 224
        )
    
    def forward(self, input):
        return self.main(input)