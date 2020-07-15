import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from Model import *

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    # Each epoch, we have to go through every data in dataset
    for epoch in range(num_epochs):
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):
            
            # initialize gradient for network
            discriminator.zero_grad()
            
            # send the data into device for computation
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            output = discriminator(real_cpu).view(-1)
            
            # Send data to discriminator and calculate the loss and gradient
            errD_real = criterion(output, label)
            errD_real.backward()
            
            # For calculate loss, you need to create label for your data
            D_x = output.mean().item()

            ## Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            
            # Send data to discriminator and calculate the loss and gradient
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            
            # For calculate loss, you need to create label for your data
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_d.step()
            
            # Update your network
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            
            # Record your loss every iteration for visualization
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()
            
            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

     
            # Remember to save all things you need after all batches finished!!!
            G_losses.append(errG.item())
            D_losses.append(errD.item())
    
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 1 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                
            iters += 1
            
    return img_list ,G_losses ,D_losses
    

if __name__ == '__main__':
    
    batch_size = 128
    image_size = 64
    
    num_epochs = 10
    lr = 0.0002
    beta1 = 0.5
    
    print('----------------data processing----------------')
    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    dataset = dset.ImageFolder(root='C:/Users/SeasonTaiInOTA/Downloads/HW3/DL_HW3/GAN_source_code',
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)
    print('----------------data processed----------------')

    real_batch = next(iter(dataloader))
#    plt.figure(figsize=(8,8))
#    plt.axis("off")
#    plt.title("Training Images")
#    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    print('----------------create generator----------------')
    # Create the generator and the discriminator()
    # Initialize them 
    # Send them to your device
    generator = Generator(ngpu).to(device)
    generator.apply(weights_init)
    #print(generator)
    
    print('----------------create discriminator----------------')
    discriminator = Discriminator(ngpu).to(device)
    discriminator.apply(weights_init)
    #print(discriminator)
    
    print('----------------create optimizer----------------')
    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    print('----------------create criterion----------------')
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0
    
    print('----------------start training----------------')
    # Start training~~
    img_list ,G_losses ,D_losses = train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs)
    print('----------------finish----------------')
    