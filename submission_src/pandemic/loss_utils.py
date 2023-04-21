import jax
import jax.numpy as jnp
import optax

################# Loss functions #################


def bce_loss(logits, targets):
  return optax.sigmoid_binary_cross_entropy(logits, targets)


def focal_loss(logits, targets, gamma=2.0, alpha=0.75):
  # References:
  # https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf
  # https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html
  # Default gamma=2, alpha=0.25 from the paper
  # Why is alpha lower instead of higher, if it applies to minority class (y=1)?
  # From the paper: "as easy negatives are downweighted (by gamma), less emphasis
  # needs to be placed on the positives"
  probs = jax.nn.sigmoid(logits)
  p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
  alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
  per_example_losses = -jnp.log(p_t) * alpha_t * (1.0 - p_t)**gamma
  return per_example_losses


def forward_bce_loss(model, params, batch):
  inputs, targets = batch
  logits = model.apply(params=params, inputs=inputs).squeeze()
  per_example_losses = bce_loss(logits, targets)
  return jnp.mean(per_example_losses)


def forward_focal_loss(model, params, batch, gamma=2.0, alpha=0.75):
  inputs, targets = batch
  logits = model.apply(params=params, inputs=inputs).squeeze()
  per_example_losses = focal_loss(logits, targets, gamma=gamma, alpha=alpha)
  return jnp.mean(per_example_losses)
