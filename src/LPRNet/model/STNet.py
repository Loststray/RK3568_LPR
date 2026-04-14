import torch.nn as nn
import torch
import torch.nn.functional as F

class STNet(nn.Module):
    
    def __init__(self):
        super(STNet, self).__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True),
                nn.Conv2d(32, 32, kernel_size=5),
                nn.MaxPool2d(3, stride=3),
                nn.ReLU(True)
                )
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
                nn.Linear(32 * 14 * 2, 32),
                nn.ReLU(True),
                nn.Linear(32, 3*2)
                )
        # Initialize the weights/bias with identity transformation 
        self.fc_loc[2].weight.data.zero_() # type: ignore
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float)) # type: ignore
        
    def forward(self, x):
        
        xs = self.localization(x)
        xs = xs.view(-1, 32*14*2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1,2,3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        
        return x

def build_STNet(phase):

    Net = STNet()

    if phase == True:
        return Net.train()
    else:
        return Net.eval()


if __name__ == "__main__":
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = STNet().to(device)
    
    input = torch.Tensor(2, 3, 24, 94).to(device)
    output = model(input)
    print('output shape is', output.shape)
