from models import generator,discriminator
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

vgg = torchvision.models.vgg19(pretrained=True)

lr = 0.0001
epochs = 1000
batch_size = 64

gen_loss = nn.BCELoss()
vgg_loss = nn.MSELoss()
mse_loss = nn.MSELoss()
disc_loss = nn.BCELoss()

gen = generator()
disc = discriminator()

gen_optimizer = optim.Adam(gen.parameters(),lr=lr)
disc_optimizer = optim.Adam(disc.parameters(),lr=lr)

for epoch in range(epochs):
    for batch in batches:

        #low res
        lr =
        #high res
        hr =

        #train discriminator

        gen_out = gen(lr)

        f_label = disc(gen_out)
        r_label = disc(hr)

        d1_loss = disc_loss(f_label,torch.zeros_like(f_label))
        d2_loss = disc_loss(r_label,torch.ones_like(r_label))

        d_loss = (d1_loss + d2_loss)/2

        disc_optimizer.zero_grad()
        d_loss.backward(retain_graph = True)
        disc_optimizer.step()

        #train generator

        m_loss = mse_loss(gen_loss,hr)
        v_loss = vgg_loss(vgg.features[:7](gen_out),vgg.features[:7](hr))
        g_loss = gen_loss(f_label,torch.ones_like(f_label))

        generator_loss = g_loss + v_loss + m_loss

        gen_optimizer.zero_grad()
        generator_loss.backward()
        gen_optimizer.step()

