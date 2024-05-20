import lightning as L
import torch
from torch import nn

from D3Fold.models.attention_transformer.causal_attention import AttentionBlock, create_causal_mask
from D3Fold.models.attention_transformer.causal_ipa import IPABlock

class GIAT(L.LightningModule):
    def __init__(
        self, 
        num_tokens,
        num_output_tokens,
        embed_dim=256, 
        num_heads=4,
        num_attention_layers=4,
        num_ipa_layers=4,
        pad_value=-100,
    ):
        super(GIAT, self).__init__()
        self.embedder = nn.Embedding(num_tokens, embed_dim)
        self.pad_value = pad_value
        
        self.final_cos_phi = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_phi = nn.Linear(embed_dim, num_output_tokens)
        self.final_cos_psi = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_psi = nn.Linear(embed_dim, num_output_tokens)
        self.final_cos_omega = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_omega = nn.Linear(embed_dim, num_output_tokens)

        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(num_tokens, embed_dim, num_heads, hidden_size=256)
                for _ in range(num_attention_layers)
            ]
        )
        self.ipa_blocks = nn.ModuleList(
            [
                IPABlock(
                    dim = 256,                  # single (and pairwise) representation dimension
                    heads = 8,                 # number of attention heads
                    scalar_key_dim = 16,       # scalar query-key dimension
                    scalar_value_dim = 16,     # scalar value dimension
                    point_key_dim = 4,         # point query-key dimension
                    point_value_dim = 4,        # point value dimension
                    require_pairwise_repr = False
                )
                for _ in range(num_ipa_layers)
            ]
        )

    def forward(self, data):
        x = data.aatype
        x = x.masked_fill(x == self.pad_value, self.embedder.num_embeddings - 1)
        x = self.embedder(x)
        mask = create_causal_mask(x.size(1), x.device)

        for block in self.attention_blocks:
            x = block(x, mask)

        rotations = data.backbone_rigid_tensor[...,:3, :3]
        translations = data.backbone_rigid_tensor[...,:3, 3]

        for block in self.ipa_blocks:
            x = block(
                x,
                rotations = rotations,
                translations = translations,
                mask = mask              
            )

        cos_pred_phi = self.final_cos_phi(x)
        sin_pred_phi = self.final_sin_phi(x)
        
        cos_pred_psi = self.final_cos_psi(x)
        sin_pred_psi = self.final_sin_psi(x)

        cos_pred_omega = self.final_cos_omega(x)
        sin_pred_omega = self.final_sin_omega(x)

        phi = torch.stack([cos_pred_phi, sin_pred_phi], dim=-1)
        psi = torch.stack([cos_pred_psi, sin_pred_psi], dim=-1)
        omega = torch.stack([cos_pred_omega, sin_pred_omega], dim=-1)
        
        return phi, psi, omega
    
    def training_step(self, batch, batch_idx):
        cross = nn.CrossEntropyLoss()
        phi_pred, psi_pred, omega_pred = self(batch)
        phi_pred = phi_pred.permute(0,2,1,3)
        psi_pred = psi_pred.permute(0,2,1,3)
        omega_pred = omega_pred.permute(0,2,1,3)
        phi = batch.quantized_phi_psi_omega[:,:,0,:]
        psi = batch.quantized_phi_psi_omega[:,:,1,:]
        omega = batch.quantized_phi_psi_omega[:,:,2,:]

        phi[torch.isnan(phi)] = -100
        psi[torch.isnan(psi)] = -100
        omega[torch.isnan(omega)] = -100

        phi_loss = cross(phi_pred, phi.long())
        psi_loss = cross(psi_pred, psi.long())
        omega_loss = cross(omega_pred, omega.long())
        full_loss = phi_loss + psi_loss + omega_loss
        self.log("loss", full_loss)
        return full_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer