# Utils
from models import Discriminator, Generator
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} as device...")


noise_size = 64
batch_size = 8
gradient_accumulations = 4

test_noise = torch.rand((batch_size, noise_size)).to(device)

epochs = 400
k = 1 #hyperparameter from the original paper (The number of steps to apply to the discriminator)

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

MNIST_dataset = datasets.MNIST(root='./', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(MNIST_dataset, batch_size=batch_size, shuffle=True)
loss_func  = nn.BCEWithLogitsLoss() # loss(x,y) = - (y * log(x) + (1-y) * log(1 - x))

discriminator = Discriminator(img_features=28*28).to(device)
generator = Generator(noise_size=noise_size, img_features=28*28).to(device)

disc_optimizer = optim.Adam(discriminator.parameters(), lr=3e-4)
gen_optimizer = optim.Adam(generator.parameters(), lr=3e-4)



fake_log = SummaryWriter(f"logs/fake")
real_log = SummaryWriter(f"logs/real")
step_ = 0

scaler = GradScaler()

def train():
    global step_
    #the GAN Training Algorithm from the original paper
    for epoch in tqdm(range(epochs)):
        #The training of the discriminator 
        for batch_idx, (real_sample,_) in enumerate(dataloader): 
            for _ in range(k): #if k=1, this for should be commented

                with autocast():
                  noise_sample = torch.rand((batch_size, noise_size)).to(device)
                  real_sample = real_sample.to(device)
                  fake_sample = generator(noise_sample)
                  #Discriminator loss: min -log(D(real) + log(1 - D(G(noise))))
                  real_distribution = discriminator(real_sample)
                  fake_distribution = discriminator(fake_sample)
                  loss_real = loss_func(real_distribution, torch.ones_like(real_distribution))
                  loss_fake = loss_func(fake_distribution, torch.zeros_like(fake_distribution))
                  disc_loss = (loss_real + loss_fake)/2

                if (batch_idx + 1) % gradient_accumulations == 0:
                  scaler.scale(disc_loss / gradient_accumulations).backward(retain_graph=True)
                  scaler.step(disc_optimizer)
                  scaler.update()
                
        
            #The trianing of the generator
            #Generator loss:  min log(1-D(G(noise)) <=> min -log(D(G(noise))), to avoid gradient saturation
            with autocast():
              distribution = discriminator(fake_sample)
              gen_loss = loss_func(distribution, torch.ones_like(distribution))
              
              if (batch_idx + 1) % gradient_accumulations == 0:
                scaler.scale(gen_loss / gradient_accumulations).backward()
                scaler.step(gen_optimizer)
                scaler.update()

            #memory cleaning
            gen_loss = gen_loss.detach().cpu().numpy()
            disc_loss = disc_loss.detach().cpu().numpy()

            if batch_idx == 0:
                print(f"Generator Loss: {gen_loss}, Discriminator_loss: {disc_loss} on epoch {epoch}")
                with torch.no_grad():
                    fake_sample = generator(test_noise).reshape(-1, 1, 28, 28)

                    fake_grid = torchvision.utils.make_grid(fake_sample, normalize=True)
                    data_grid = torchvision.utils.make_grid(real_sample, normalize=True) 


                fake_log.add_image('Generated Fake Digits', fake_grid, global_step=step_)
                fake_log.add_image('Real Digits', data_grid, global_step=step_)

                step_+=1
            
            #meomory cleaning
            noise_sample = noise_sample.detach().cpu().numpy()
            real_sample = real_sample.detach().cpu().numpy()
            fake_sample = fake_sample.detach().cpu().numpy()
            loss_real = loss_real.detach().cpu().numpy()
            loss_fake = loss_fake.detach().cpu().numpy()
            


train()
