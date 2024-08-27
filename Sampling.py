import torch
from typing import Dict
from ModelConfig import modelConfig
from torchvision.utils import save_image

from UNet import UNet
from Diffusion import GaussianDiffusionSampler

def sampling(modelConfig: Dict):
    with torch.no_grad():
        device = modelConfig['device']
        model = UNet(T=modelConfig['T'], ch=modelConfig['channel'], ch_mult=modelConfig['channel_mult'], attn=modelConfig['attn'], num_res_blocks=modelConfig['num_res_blocks'], dropout=modelConfig['dropout']).to(device).float()
        checkpoint = torch.load(modelConfig['checkpoint_dir'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        sampler = GaussianDiffusionSampler(model=model,
                                           beta_1=modelConfig['beta_1'],
                                           beta_T=modelConfig['beta_T'],
                                           T=modelConfig['T'])
        
        noisyImage = torch.randn(size=[modelConfig['batch_size'], 3, 32, 32], device=device)

        saveNoisy = (noisyImage + 1) / 2
        save_image(saveNoisy, f"{modelConfig['sample_dir']}/NoisyImgs.png")

        sampledImage = sampler(noisyImage)

        saveSampled = (sampledImage + 1) / 2
        save_image(saveSampled, f"{modelConfig['sample_dir']}/sampledImgs.png")


if __name__ == "__main__":
    sampling(modelConfig)