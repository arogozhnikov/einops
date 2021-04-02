from einops.einattention import EinAttentionRecipe
import torch

recipe = EinAttentionRecipe('t s * head <- t (head *) s, t (head *) s2', head=2)
result = recipe.forward(torch.ones(3, 4, 5), torch.ones(3, 4, 6), torch.ones(3, 8, 6))
print(result.shape)
print(result)