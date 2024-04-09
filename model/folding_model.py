import torch
from torch import nn
from mamba_ssm import Mamba

import esm
from torch_geometric.nn import radius_graph
import lightning as L
import numpy as np

from model.losses import sequence_loss

class FoldingTrunk(nn.Module):
    def __init__(self, s_dim_in=1280, s_dim_out=32, z_dim_in=1,z_dim_out=32):
        super().__init__()
        self.s_project = nn.Linear(s_dim_in, s_dim_out)
        self.z_project = nn.Linear(z_dim_in, z_dim_out)
        # this is black magic rn, I have no idea what Mamba does
        self.mamba = Mamba(
            d_model=s_dim_out, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def outer_product(self, x, y):
      return torch.einsum("bsf,btf->bstf", x, y)

    def forward(self, s, z):
        s_prime = self.s_project(s)
        z_prime = self.z_project(z)
        z_prime = z_prime + self.outer_product(s_prime, s_prime)
        return s_prime, z_prime


class IPA(nn.Module):
    def __init__(self, s_dim_in=32, s_dim_out=32, z_dim_in=32, z_dim_out=32):
        super().__init__()


class D3Fold(L.LightningModule):
    def __init__(self, mamba_layers=3, freeze_esm=False):
        super().__init__()
        self.losses = [sequence_loss]
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = model
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)
        esm_dim = 1280
        self.folding_trunk = FoldingTrunk(
            s_dim_in=esm_dim,
            s_dim_out=32,
            z_dim_in=1,
            z_dim_out=32,
        )

        self.final_z_proj = nn.Linear(32, 1)
        self.final_s_proj = nn.Linear(32, 21)

    @staticmethod
    def get_distance_matrix(data, r=10):
      num_graphs = data.seq.shape[0]
      contact_edges = radius_graph(data.coords, r=r,  batch=data.batch)
      graph_sizes = data.batch.bincount()
      largest_graph = torch.max(graph_sizes)
      num_nodes = data.coords.shape[0]
      contact_mat = torch.zeros((num_nodes, num_nodes))
      src_indices, tgt_indices = contact_edges[0], contact_edges[1]
      contact_mat[src_indices, tgt_indices] = 1
      contact_mat[tgt_indices, src_indices] = 1
      sequence_of_contacts = torch.zeros((num_graphs, largest_graph, largest_graph))
      current_graph_index = 0
      for i in range(num_graphs):
          contacts = contact_mat[current_graph_index:current_graph_index+graph_sizes[i], current_graph_index:current_graph_index+graph_sizes[i]]
          contacts += torch.diag(torch.ones(contacts.shape[0]))

          # if shape is less than max pad with zeros
          if contacts.shape[0] < largest_graph:
              contacts = torch.nn.functional.pad(contacts, (0, largest_graph - contacts.shape[0], 0, largest_graph - contacts.shape[1]))

          sequence_of_contacts[i] = contacts
          current_graph_index += graph_sizes[i]

      return sequence_of_contacts

    @staticmethod
    def get_distance_mat_stack(data, min_radius=5, max_radius=20, num_radii=4):
        radius_list = np.linspace(min_radius, max_radius, num_radii)
        stack = []
        for r in radius_list:
            mat = D3Fold.get_distance_matrix(data, r=r)
            stack.append(mat)

        return torch.stack(stack, dim=-1)

    def forward(self, batch):
        esm_seq = batch.tokens
        seq_embed = self.esm(esm_seq.long(), repr_layers=[33], return_contacts=True)
        # attention based contacts
        att_contacts = seq_embed['contacts'].unsqueeze(-1)
        rep = seq_embed['representations'][33][:,1:-1]
        s, z = self.folding_trunk(rep, att_contacts)
        z = self.final_z_proj(z).squeeze(-1)
        s = self.final_s_proj(s)
        s = s.softmax(dim=-1)
        att_contacts = att_contacts.squeeze(-1)
        return s, z

    def training_step(self, batch, batch_idx):
        s, z = self.forward(batch)
        distance_mats = self.get_distance_mat_stack(batch)
        loss = 0
        for loss_fn in self.losses:
            # this is shit but on right track
            if loss_fn.representation_target == "seq":
                loss += loss_fn(s, batch.seq, mask=batch.mask)
            if loss_fn.representation_target == "pair":
                loss += loss_fn(z, distance_mats)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
