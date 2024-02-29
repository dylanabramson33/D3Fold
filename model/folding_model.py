import torch
from torch import nn
from mamba_ssm import Mamba
from torch.nn import functional as F

import esm
from torch_geometric.nn import radius_graph


class PairRepModule(nn.Module):
    def __init__(self):
        super().__init__()

    def outer_product(self, x, y):
        return torch.einsum("bi,bj->bij", x, y)

    def forward(self, x):
        return x

class FoldingTrunk(L.LightningModule):
    def __init__(self, mamba_layers=3, freeze_esm=False):
        super().__init__()

        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = model.cuda()
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)
        

        esm_dim = 1280
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1280, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        ).to("cuda")

    @staticmethod
    def get_distance_matrix(data):
      num_graphs = data.seq.shape[0]
      contact_edges = radius_graph(data.coords, r=5,  batch=data.batch)
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
              # add in diagonal
          sequence_of_contacts[i] = contacts
          current_graph_index += graph_sizes[i]

      return sequence_of_contacts

    def make_sequence_dense(self, seq):
        seq_len = seq.shape[0]
        dense_seq = torch.zeros((seq_len, seq_len))

    def forward(self, batch):
        coords = batch.coords
        esm_seq = batch.tokens
        seq_embed = self.esm(esm_seq.long(), repr_layers=[33], return_contacts=True)
        files = batch.file
        pred_contacts = seq_embed['contacts']
        pred_contacts = pred_contacts
        rep = seq_embed['representations'][33]
        mamba = self.mamba(rep)
        # pred_contacts = self.mamba(rep)
        return pred_contacts

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch.cuda())
        distance_mat = self.get_distance_matrix(batch)

        loss = F.binary_cross_entropy(pred, distance_mat.cuda())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
