import torch
import math
from torch import nn
from mamba_ssm import Mamba

import esm
import lightning as L

from d3fold.data.openfold.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from d3fold.model.losses import pairwise_loss, seq_loss
from invariant_point_attention import InvariantPointAttention
from d3fold.model import utils as model_utils
from d3fold.model.common.positional_encoding import PositionalEncoding


class ESMEmbedder(nn.Module):
    def __init__(self, s_dim_in=1280, s_dim_out=32, z_dim_in=1,z_dim_out=32, freeze_esm=False):
        super().__init__()
        self.esm, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = self.esm.cuda()
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)

        self.s_project = nn.Sequential(
            nn.Linear(s_dim_in, s_dim_out),
            nn.ReLU(),
            nn.Linear(s_dim_out, s_dim_out),
        )
        self.position_encoder = PositionalEncoding(s_dim_out)

        self.z_project = nn.Sequential(
            nn.Linear(z_dim_in, z_dim_out),
            nn.ReLU(),
            nn.Linear(s_dim_out, s_dim_out),
        )

    def forward(self, batch):
        esm_seq = batch.tokens
        seq_embed = self.esm(esm_seq.long(), repr_layers=[33], return_contacts=True)
        # attention based contacts
        z = seq_embed['contacts'].unsqueeze(-1)
        s = seq_embed['representations'][33][:,1:-1]
        s = self.s_project(s)
        s = s + self.position_encoder(batch)
        z = self.z_project(z)
        return s, z

class MambaBlock(nn.Module):
    def __init__(self, hidden_size=128):
      super().__init__()
      self.block = nn.Sequential(
        Mamba(hidden_size),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.ReLU(),
      ).cuda()

    def forward(self, s):
        return self.block(s)

class EvoMamba(nn.Module):
    def __init__(self, hidden_size=128, num_layers=24):
        super().__init__()

        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(hidden_size) for _ in range(num_layers)]
        )

    def outer_product(self, x, y):
        return torch.einsum("bsf,btf->bstf", x, y)

    def forward(self, s, z):
        for block in self.mamba_blocks:
            mamba_op = block(s)
            s = s + mamba_op
            z = z + self.outer_product(s, s)

        return s, z

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
                 mambaformer_block_layers=24,
                 hidden_size=128,
                 freeze_esm=True,
                 n_dist_bins=65,
                 ):
        super().__init__()

        self.losses = [pairwise_loss, seq_loss]
        esm_dim = 1280

        self.esm_embedder = ESMEmbedder(
            s_dim_in=esm_dim,
            s_dim_out=hidden_size,
            z_dim_in=1,
            z_dim_out=hidden_size,
            freeze_esm=freeze_esm
        )

        self.mambaformer = EvoMamba(hidden_size=hidden_size, num_layers=mambaformer_block_layers)

        self.final_z_proj = nn.Linear(hidden_size, n_dist_bins)
        self.final_s_proj = nn.Linear(hidden_size, 21)

    def forward(self, batch):
        s, z = self.esm_embedder(batch)
        s, z = self.mambaformer(s, z)
        z = self.final_z_proj(z)
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
            loss += loss_component
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )

        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )


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
