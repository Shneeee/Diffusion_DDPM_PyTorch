import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(v, t, x_shape):
  '''
  v: Value corresponding to each timestep
  t: Tensor with randomly selected timestep index

  Use t as an index to extract values from the vector v, and reshape it to match the dimensions of x_shape.
  [batch_size, 1, 1, 1, ...]
  '''
  v = v.to(t.device)
  out = torch.gather(v, index=t, dim=0).float()
  return out.view([t.shape[0]] + [1] * (len(x_shape) - 1)).to(t.device)

class GaussianDiffusionTrainer(nn.Module):
  def __init__(self, model, beta_1, beta_T, T):
    super().__init__()
    self.model = model
    self.T = T
    self.register_buffer(
        'betas', torch.linspace(beta_1, beta_T, T).double())
    alphas = 1. - self.betas 
    alphas_bar = torch.cumprod(alphas, dim=0) 

    # Calculate for diffusion process q(x_t | x_0)
    self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar)) 
    self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

  def forward(self, x_0):
    '''
    In the diffusion process, the noisy images x_t at a specific time t is generated as a combination of the original images x_0 and Gaussian noise ϵ,
    and the loss is calculated as the 'MSE loss' between the noise predicted by the model and the actual noise.
    '''
    t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
    noise = torch.randn_like(x_0) 

    # x_t = √α̅_t * x_0 + √(1 - α̅_t)ϵ
    x_t = (
        extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
        extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
    
    loss = F.mse_loss(self.model(x_t, t), noise, reduction='mean')
    return t, noise, x_t, loss
  
class GaussianDiffusionSampler(nn.Module):
  def __init__(self, model, beta_1, beta_T, T, img_size=32, mean_type='eps', var_type='fixedlarge'):
    super().__init__()
    self.model = model
    self.T = T
    self.img_size = img_size
    self.mean_type = mean_type
    self.var_type = var_type

    self.register_buffer(
        'betas', torch.linspace(beta_1, beta_T, T).double())
    alphas = 1. - self.betas 
    alphas_bar = torch.cumprod(alphas, dim=0) 
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

    self.register_buffer(
        'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
    self.register_buffer(
        'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))


    self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
    self.register_buffer('posterior_log_var_clipped', torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
    self.register_buffer('posterior_mean_coef1', torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
    self.register_buffer('posterior_mean_coef2', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))
  
  # Calculate the mean and log variance of q(x_{t-1} | x_t, x_0)
  def q_mean_variance(self, x_0, x_t, t):
    posterior_mean = (
        extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
        extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, x_t.shape)
    return posterior_mean, posterior_log_var_clipped
  
  # Estimate x_0 using noise prediction
  def predict_x_start_from_eps(self, x_t, t, eps):
    return (
        extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
    )
  
  # Estimate x_0 from samples from previous steps
  def predict_x_start_from_xprev(self, x_t, t, xprev):
    return (
        extract(
            1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
        extract(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
    )
  
  # Calculate the mean and log variance of the x_{t-1} at x_t and timestep t
  def p_mean_variance(self, x_t, t):
    model_log_var = {
        'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2], self.betas[1:]])),
        'fixedsmall': self.posterior_log_var_clipped,
    }[self.var_type]
    model_log_var = extract(model_log_var, t, x_t.shape)

    if self.mean_type == 'xprev':
      x_prev = self.model(x_t, t)
      x_0 = self.predict_x_start_from_xprev(x_t, t, x_prev=x_prev)
      model_mean = x_prev
    elif self.mean_type == 'xstart':
      x_0 = self.model(x_t, t)
      model_mean, _ = self.q_mean_variance(x_0, x_t, t)
    elif self.mean_type == 'eps':
      eps = self.model(x_t, t)
      x_0 = self.predict_x_start_from_eps(x_t, t, eps=eps) 
      model_mean, _ = self.q_mean_variance(x_0, x_t, t) 
    else:
      raise NotImplementedError(self.mean_type)

    x_0 = torch.clip(x_0, -1., 1.)

    return model_mean, model_log_var

  def forward(self, x_T):
    '''
    In the reverse process, the state {x_(t-1)} at the previous timestep is estimated based on the state x_t at the current timestep,
    which is repeatedly denoised until it is restored to the original data x_0.
    '''
    x_t = x_T
    print(f'Sampler Input x_t {x_T.shape}')
    batch_size = x_T.shape[0]

    for time_step in reversed(range(self.T)):
      t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step
      mean, log_var = self.p_mean_variance(x_t=x_t, t=t)

      if time_step > 0:
        noise = torch.randn_like(x_t)
      else:
        noise = 0

      # x_{t-1} = μθ(x_t, t) + σ_t * z
      x_t = mean + torch.exp(0.5 * log_var) * noise 
    x_0 = x_t
    return torch.clip(x_0, -1, 1)
