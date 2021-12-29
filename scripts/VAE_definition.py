import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(nn.Module):
    def __init__(self, 
                 latent_dim :int, 
                 image_size:int,
                 encoder,
                 decoder,
                 colours:int=3,
                 hidden_layer_size:int=128):

        super(VAE, self).__init__()
        self.colours=colours
        self.image_size=image_size
        self.hidden_layer_size=hidden_layer_size
        self.latent_dim=latent_dim
        
        #Encoder layers
        self.encoder= encoder(self.colours,self.latent_dim,self.hidden_layer_size,self.image_size)
        self.decoder=decoder(self.colours,self.latent_dim,self.hidden_layer_size,self.image_size)
        self.sampling_mean=nn.Linear(self.hidden_layer_size,self.latent_dim)
        self.sampling_var=nn.Linear(self.hidden_layer_size,self.latent_dim)

    def sample(self,mean,log_var):
        std=torch.exp(0.5*log_var)
        e=torch.randn_like(mean)
        return e.mul(std).add(mean)
            

    def forward(self,x):
        x=self.encoder(x)
        mean=self.sampling_mean(x)
        var=self.sampling_var(x)

        z=self.sample(mean,var)

        x=self.decoder(z)
        return x,mean,var

def encoder_1(colours,latent_dim,hidden_layer_size,image_size):

    return (nn.Sequential(
    # shape of input is n x colours x image_h x image_w
    nn.Conv2d(colours,out_channels=32,kernel_size=8,stride=2),
    # shape n x 16 x (image_h-4)/2+1 x (image_w-4)/2+1
    nn.BatchNorm2d(32),
    nn.LeakyReLU(),
    nn.MaxPool2d(2),
    # shape n x 16 x image_h/2-1 x image_w/2-1
    
    nn.Conv2d(32, 16, 6),
    # shape n x 32 x image_h/2-1-4+1 x image_w/2-1-4+1
    nn.BatchNorm2d(16),
    nn.LeakyReLU(),
    nn.MaxPool2d(2,stride=1),
    # shape n x 8 x image_h/2-4 x image_h/2-4 

    nn.Conv2d(16, 8, 6),
    nn.Flatten(),

    nn.Linear(8*(int(image_size[0]/4)-13)*(int(image_size[0]/4)-13),hidden_layer_size),
    nn.BatchNorm1d(hidden_layer_size),
    nn.Sigmoid()))

def decoder_1(colors,latent_dim,hidden_layer_size,image_size):
    return (nn.Sequential(
            nn.Linear(latent_dim,hidden_layer_size),
            nn.Linear(hidden_layer_size,6*(int(image_size[0]/4))*(int(image_size[0]/4))),
            nn.Unflatten(1,(6,int(image_size[0]/4),int(image_size[0]/4))),

            nn.ConvTranspose2d(in_channels=6,out_channels=16,kernel_size=6, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            #image_size/4*2+4+1-2-4=image_size/2
            nn.ConvTranspose2d(in_channels=16,out_channels=colors,kernel_size=2, stride=2),
            #image_size/2*2-2+2-1+1=image_size
            ))

def decoder_2(colors,latent_dim,hidden_layer_size,image_size):
    return (nn.Sequential(
            nn.Linear(latent_dim,hidden_layer_size),
            nn.Linear(hidden_layer_size,6*(int(image_size[0]/4))*(int(image_size[0]/4))),
            nn.Unflatten(1,(6,int(image_size[0]/4),int(image_size[0]/4))),

            nn.ConvTranspose2d(in_channels=6,out_channels=64,kernel_size=6, stride=2, padding=2),
            nn.LeakyReLU(0.1),
            #image_size/4*2+4+1-2-4=image_size/2
            nn.ConvTranspose2d(in_channels=64,out_channels=colors,kernel_size=2, stride=2),
            #image_size/2*2-2+2-1+1=image_size
            ))

def simpler_encoder(colours,hidden_layer_size):
    return (nn.Sequential(
            nn.Conv2d(colours,out_channels=64,kernel_size=20,stride=2),
        # shape n x 16 x (image_h-4)/2+1 x (image_w-4)/2+1
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 32, 6),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(3,stride=1),
            nn.Flatten(),
            nn.Linear(3872,hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Sigmoid()))

def bigger_encoder(colors,latent_dim,hidden_layer_size,image_size):
    return(nn.Sequential(
        nn.Conv2d(3,out_channels=64,kernel_size=12,stride=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
        nn.MaxPool2d(2),
        # shape n x 16 x image_h/2-1 x image_w/2-1
                
        nn.Conv2d(64, 128, 6),
        # shape n x 32 x image_h/2-1-4+1 x image_w/2-1-4+1
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.MaxPool2d(2,stride=1),
        # shape n x 8 x image_h/2-4 x image_h/2-4 

        nn.Conv2d(128, 256, 6),
        nn.Flatten(),
        nn.Linear(12544,hidden_layer_size),
        nn.BatchNorm1d(hidden_layer_size),
        nn.Sigmoid()))
