import os
import pickle
from collections import defaultdict

import esm
import torch
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from protein_data import Chain 

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()

class CollateFactory():
  def __init__(self):
    self.collators = {}

  def add_collator(self, collaltion_type, collate_fn):
    self.collators[collaltion_type] = collate_fn

  def get_collate_fn(self, collaltion_type):
    return self.collators[collaltion_type]

collate_factory = CollateFactory()

def collate_seq(seqs):
  return pad_sequence([d for d in seqs], batch_first=True, padding_value=torch.nan)

def collate_torch_geometric(data):
  return Batch.from_data_list(data)
  
def collate_esm(seqs):
  seqs_w_name = [(f"protein{1}",seqs) for seq in seqs]
  return batch_converter(seqs_w_name)

collate_factory.add_collator("seq", collate_seq)
collate_factory.add_collator("esm", collate_esm)
collate_factory.add_collator("torch_geometric", collate_torch_geometric)

class SingleChainData(Dataset):
    def __init__(self, chain_dir=None, pickled_dir=None, force_process = True):
        self.chain_dir = chain_dir
        self.pickled_dir = pickled_dir
        if not os.path.exists(self.pickled_dir) or force_process:
            self.preprocess()

        self.length = len(os.listdir(self.pickled_dir))
        self.files = os.listdir(self.pickled_dir)

    def preprocess(self):
        os.makedirs(self.pickled_dir, exist_ok=True)
        for i, file in enumerate(os.listdir(self.chain_dir)):
            if i > 5:
              break
            if file.endswith(".ent") and "pdb1kka" not in file:
                chain = Chain.from_pdb(os.path.join(self.chain_dir, file))
                with open(os.path.join(self.pickled_dir, file.replace(".ent", ".pkl")), "wb") as f:
                  pickle.dump(chain, f)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        with open(os.path.join(self.pickled_dir, f), "rb") as f:
            chain = pickle.load(f)

        data_fields = sorted(list(chain.__dataclass_fields__.keys()))
        chain.mask_data()
        collation_dict = defaultdict(list)
        for field in data_fields:
            field_data = getattr(chain, field)
            collation_dict[field_data.type_.collate_type].append(field_data)

        return collation_dict

def unpack_collation_dicts(collation_dicts, collation_type):
  collation_datalist = [collation_dict[collation_type] for collation_dict in collation_dicts]
  type_to_datalist = defaultdict(list)
  for collation_data in collation_datalist:
    for field_data in collation_data:
      type_to_datalist[field_data.type_.type].append(field_data.data)

  return type_to_datalist

def collate_chains(collation_dicts):
    data = Data()
    for collate_type in collation_dicts[0].keys():
        type_to_datalist = unpack_collation_dicts(collation_dicts, collate_type)
        for type_, datalist in type_to_datalist.items():
            collate_fn = collate_factory.get_collate_fn(collate_type)
            data[type_] = collate_fn(datalist)
         
    return data