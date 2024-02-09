import os
import pickle

import esm
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from protein_data import Chain 

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

class SingleChainData(Dataset):
    def __init__(self, chain_dir=None, pickled_dir=None, force_process = True, limit_by=None):
        self.chain_dir = chain_dir
        self.pickled_dir = pickled_dir
        self.limit_by = limit_by
        if not os.path.exists(self.pickled_dir) or force_process:
            self.preprocess()

        self.length = len(os.listdir(self.pickled_dir))
        self.files = os.listdir(self.pickled_dir)
       

    def preprocess(self):
        os.makedirs(self.pickled_dir, exist_ok=True)
        for i, file in enumerate(os.listdir(self.chain_dir)):
            if self.limit_by and i > self.limit_by:
              break
            if file.endswith(".ent"):
              try:
                chain = Chain.from_pdb(os.path.join(self.chain_dir, file))
                with open(os.path.join(self.pickled_dir, file.replace(".ent", ".pkl")), "wb") as f:
                  pickle.dump(chain, f)
              except Exception as e:
                print(f"Failed to load {file}")
                print(e)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        with open(os.path.join(self.pickled_dir, f), "rb") as f:
            chain = pickle.load(f)

        data_fields = list(chain.__dataclass_fields__.keys())
        chain.mask_data()
        geo_data = {}
        seq_data = {}
        raw_seq_data = {}
        for field in data_fields:
            field_data = getattr(chain, field)
            if field_data.type_.pad_type == "torch_geometric":
                geo_data[field] = field_data.data
            elif field_data.type_.pad_type == "seq":
                seq_data[field] = field_data.data
            elif field_data.type_.pad_type == "esm":
                raw_seq_data[field] = field_data.data

        geo_data = Data.from_dict(geo_data)
        geo_data.file = f
        return geo_data, seq_data, raw_seq_data

def collate_chains(data_list):
    geo_data_list = [d[0] for d in data_list]
    seq_data_list = [d[1] for d in data_list]
    raw_seq_data_list = [(f"protein{i}", ''.join(d[2]['raw_seq'])) for i,d in enumerate(data_list)]
    batch_data = Batch.from_data_list(geo_data_list)
    for key in seq_data_list[0].keys():
        seq = pad_sequence([d[key] for d in seq_data_list], batch_first=True, padding_value=torch.nan)
        batch_data[key] = seq
    _, _, batch_tokens = batch_converter(raw_seq_data_list)
    batch_data.tokens = batch_tokens
        
    return batch_data