import torch 

modelConfig = {
    'epoch': 50,
    'batch_size': 128,
    'T': 1000,
    'channel': 128,
    'channel_mult': [1, 2, 3, 4],
    'attn': [2],
    'num_res_blocks': 2,
    'dropout': 0.1,
    'beta_1': 1e-4,
    'beta_T': 0.02,
    'lr': 2e-4,
    'checkpoint_dir':'./checkpoints',
    'noisy_data_dir':'/noisydata',
    'sampling_dir': '/samplingdata',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'img_size': 32,
    'nrow': 8
}
