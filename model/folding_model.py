import torch
from torch import nn
from mamba_ssm import Mamba

import esm
import lightning as L

from D3Fold.model.losses import pairwise_loss, seq_loss
from invariant_point_attention import InvariantPointAttention
from D3Fold.model import utils as model_utils

class FoldingBlockESM(nn.Module):
    def __init__(self, s_dim_in=1280, s_dim_out=32, z_dim_in=1,z_dim_out=32, freeze_esm=False):
        super().__init__()
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = model
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)
        self.s_project = nn.Sequential(
            nn.Linear(s_dim_in, s_dim_out),
            nn.ReLU(),
            nn.Linear(s_dim_out, s_dim_out),
        )
        self.z_project = nn.Sequential(
            nn.Linear(z_dim_in, z_dim_out),
            nn.ReLU(),
            nn.Linear(s_dim_out, s_dim_out),
        )
        # this is black magic rn, I have no idea what Mamba does
        self.mamba = Mamba(
            d_model=s_dim_out, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def outer_product(self, x, y):
      return torch.einsum("bsf,btf->bstf", x, y)

    def forward(self, batch):
        esm_seq = batch.tokens
        seq_embed = self.esm(esm_seq.long(), repr_layers=[33], return_contacts=True)
        # attention based contacts
        z = seq_embed['contacts'].unsqueeze(-1)
        s = seq_embed['representations'][33][:,1:-1]
        s_prime = self.s_project(s)
        z_prime = self.z_project(z)
        z_prime = z_prime + self.outer_product(s_prime, s_prime)
        return s_prime, z_prime


class IPA(nn.Module):
    def __init__(self, dim=32):
        self.ipa = InvariantPointAttention(
            dim=dim,
            heads=8,
            scalar_key_dim=16,
            scalar_value_dim=16,
            point_key_dim=4,
            point_value_dim=4
        )

    def forward(self, s, z, data):
        return self.ipa(s, z, rotations=data.frames_R, translations=data.frames_t, mask=data.mask)

class D3Fold(L.LightningModule):
    def __init__(self,
                 folding_block_layers=1,
                 hidden_size=128,
                 freeze_esm=False):
        super().__init__()
        self.losses = [pairwise_loss, seq_loss]

        esm_dim = 1280

        self.first_folding_block = FoldingBlockESM(
            s_dim_in=esm_dim,
            s_dim_out=hidden_size,
            z_dim_in=1,
            z_dim_out=hidden_size,
            freeze_esm=freeze_esm
        )

        self.final_z_proj = nn.Linear(hidden_size, 1)
        self.final_s_proj = nn.Linear(hidden_size, 21)

    def forward(self, batch):
        s, z = self.folding_trunk(batch)
        z = self.final_z_proj(z).squeeze(-1)
        s = self.final_s_proj(s)
        return s, z

    def training_step(self, batch, batch_idx):
        s, z = self.forward(batch)
        loss = 0
        for loss_fn in self.losses:
            # this is shit but on right track
            if loss_fn.representation_target == "seq":
                loss_component = loss_fn(s, batch)
            if loss_fn.representation_target == "pair":
                loss_component = loss_fn(z, batch)
            self.log(loss_fn.name, loss_component)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return model_utils.torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.dtype, r.device)
        return model_utils.frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
