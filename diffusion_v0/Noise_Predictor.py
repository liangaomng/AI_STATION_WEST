
from diffusers import DDPMScheduler, UNet2DModel
import seaborn as sns
import torchvision
from utils.wgan_data import *
#如何引用全部的
import torch
n_epochs=100
x,y=next(iter(ode_dataloader))



class BasicNN(nn.Module):
    '''
    输入和输出是一样的大小
    '''
    def __init__(self):
        super(BasicNN, self).__init__()
        self.hiddens= torch.nn.ModuleList([
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            ]
        )
    def forward(self, x):
        for layer in self.hiddens:
            x=layer(x)
        return x
def corrupt(x, amount):
    """Corrupt the input `x` by mixing it with noise according to `amount`"""
    noise = torch.rand_like(x)
    print("noise",noise.shape)
    amount = amount.view(-1, 1, 1)  # Sort shape so broadcasting works
    y=x * (1 - amount) + noise * amount
    print("noisy",y.shape)
    return y



if __name__ == "__main__":

    net = BasicNN().to("cuda")
    x=x.reshape(-1,300).float().to("cuda")
    print("input",x.shape)
    y = net(x)
    print(y.shape)
    #loss function
    loss_fn= nn.MSELoss()
    #optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    device="cuda"
    losses=[]
    n_epochs=100
    for epoch in range(n_epochs):
        for x,y in ode_dataloader:
            x=x.to(device)
            noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
            noisy_x = corrupt(x, noise_amount) # Create our noisy x
            noisy_x = noisy_x.reshape(-1,300).float().to("cuda")
            pred=net(noisy_x)
            x=x.reshape(-1,300).float().to("cuda")
            loss=loss_fn(pred,x)
            #反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #打印
            losses.append(loss.item())
        # Print our the average of the loss values for this epoch:
        avg_loss = sum(losses[-len(ode_dataloader):]) / len(ode_dataloader)
        print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')
    print("loss",len(losses))

    print(ode_dataloader)
    plt.plot(losses)
    plt.show()




