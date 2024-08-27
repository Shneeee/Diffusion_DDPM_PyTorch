import torch
import torch.optim as optim
from typing import Dict
from tqdm import tqdm
import os

from UNet import UNet
from Diffusion import GaussianDiffusionTrainer
from ModelConfig import modelConfig
from Dataset import get_dataloader

def train(modelConfig: Dict):
  dataloader = get_dataloader(batch_size=128, shuffle=True)
  device = torch.device(modelConfig['device'])

  model = UNet(T=modelConfig['T'], ch=modelConfig['channel'], ch_mult=modelConfig['channel_mult'], attn=modelConfig['attn'], num_res_blocks=modelConfig['num_res_blocks'], dropout=modelConfig['dropout']).to(device).float()
  optimizer = optim.Adam(model.parameters(), lr=modelConfig['lr'])
  trainer = GaussianDiffusionTrainer(model=model, beta_1=modelConfig['beta_1'], beta_T=modelConfig['beta_T'], T=modelConfig['T'])

  for epoch in range(modelConfig['epoch']):
    epoch_loss = 0.0
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch + 1}/{modelConfig["epoch"]}', unit='batch') as pbar:
      for batch_idx, (images, _) in enumerate(dataloader):
        optimizer.zero_grad()
        x_0 = images.to(device)
        t, noise, x_t, loss = trainer(x_0)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)

    epoch_loss /= len(dataloader)
    print(f'Epoch {epoch + 1}/{modelConfig["epoch"]}, Loss: {epoch_loss:.4f}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    },os.path.join(modelConfig['checkpoint_dir'], f'ckpt_{epoch + 1}.pt'))


if __name__ == "__main__":
  train(modelConfig)