import lightning as L
import torch
from torch import nn

from D3Fold.models.common.positional_encoding import PositionalEncoding
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
        self.lang_embedder = nn.Embedding(num_tokens, embed_dim)
        self.angle_embedder = nn.Embedding(num_output_tokens, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=10000)
        self.pad_value = pad_value

        # gonna refactor this to be one weight matrix
        self.final_cos_phi = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_phi = nn.Linear(embed_dim, num_output_tokens)
        self.final_cos_psi = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_psi = nn.Linear(embed_dim, num_output_tokens)
        self.final_cos_omega = nn.Linear(embed_dim, num_output_tokens)
        self.final_sin_omega = nn.Linear(embed_dim, num_output_tokens)

        self.final_lang = nn.Linear(embed_dim, num_tokens)

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
        phis = data.quantized_phi_psi_omega[:,:,1]
        psis = data.quantized_phi_psi_omega[:,:,2]
        omegas = data.quantized_phi_psi_omega[:,:,0]

        phis = self.angle_embedder(phis)
        psis = self.angle_embedder(psis)
        omegas = self.angle_embedder(omegas)

        x = x.masked_fill(x == self.pad_value, self.lang_embedder.num_embeddings - 1)
        x = self.lang_embedder(x)
        x = x + self.positional_encoding(data)
        mask = create_causal_mask(x.size(1), x.device)
        valid_frame_mask = data.backbone_rigid_mask.bool()
        for block in self.attention_blocks:
            x = block(x, mask)


        rotations = data.backbone_rigid_tensor[...,:3, :3]
        translations = data.backbone_rigid_tensor[...,:3, 3]
        for block in self.ipa_blocks:
            x = block(
                x,
                rotations = rotations,
                translations = translations,
                causal_mask = mask,
                null_mask = valid_frame_mask
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

        pred_aa = self.final_lang(x)

        return phi, psi, omega, pred_aa

    def training_step(self, batch, batch_idx):
        cross = nn.CrossEntropyLoss()
        phi_pred, psi_pred, omega_pred, pred_aa = self(batch)
        phi_pred = phi_pred.permute(0,2,1,3)
        psi_pred = psi_pred.permute(0,2,1,3)
        omega_pred = omega_pred.permute(0,2,1,3)
        pred_aa = pred_aa.permute(0,2,1)

        aa = batch.aatype.long()
        aa = aa.masked_fill(aa == self.pad_value, -100)
        aa = aa.masked_fill(aa == 20, -100)
        aa = aa.roll(-1,dims=1)

        phi = batch.quantized_phi_psi_omega[:,:,0,:].roll(-1,dims=1)
        psi = batch.quantized_phi_psi_omega[:,:,1,:].roll(-1,dims=1)
        omega = batch.quantized_phi_psi_omega[:,:,2,:].roll(-1,dims=1)

        phi_mask = batch.torsion_angles_mask[0,:,0, None].bool()
        psi_mask = batch.torsion_angles_mask[0,:,1, None].bool()
        omega_mask = batch.torsion_angles_mask[0,:,2, None].bool()
        phi = torch.where(phi_mask, phi, -100)
        psi = torch.where(psi_mask, psi, -100)
        omega = torch.where(omega_mask, omega, -100)
        phi[:, -1] = -100
        psi[:, -1] = -100
        omega[:, -1] = -100

        phi_loss = cross(phi_pred, phi.long())
        psi_loss = cross(psi_pred, psi.long())
        omega_loss = cross(omega_pred, omega.long())

        lang_loss = cross(pred_aa, aa.long())

        angle_loss = phi_loss + psi_loss + omega_loss
        full_loss = angle_loss + lang_loss

        self.log("angle_loss", angle_loss)
        self.log("lang_loss", lang_loss)
        self.log("full_loss", full_loss)
        return full_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer