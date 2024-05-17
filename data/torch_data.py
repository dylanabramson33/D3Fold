import os
import pickle

import esm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from D3Fold.data.protein import TorchProtein, ProteinData, ProteinDataType
import torch.nn.functional as F
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
FLOAT_TYPES = [torch.float32, torch.float64]
INT_TYPES = [torch.int32, torch.int64]


def pad_seqrep(list_of_tensors, key):
    if "mask" in key:
        seq = pad_sequence(
           list_of_tensors, batch_first=True, padding_value=False
        )
    elif list_of_tensors[0].dtype in FLOAT_TYPES:
        seq = pad_sequence(
            list_of_tensors, batch_first=True, padding_value=torch.nan
        )
    elif list_of_tensors[0].dtype in INT_TYPES:
        seq = pad_sequence(
            list_of_tensors, batch_first=True, padding_value=-100
        )
    
    return seq

def pad_pairrep(list_of_tensors, pad_value=torch.nan):
    maximum_residues = max(tensor.shape[0] for tensor in list_of_tensors)
    padded_tensors = []
    
    for tensor in list_of_tensors:
        # Pad the tensor to make it of size [P, P, S]
        pad_width = (0, 0, 0, maximum_residues - tensor.shape[0], 0, maximum_residues - tensor.shape[1])
        padded_tensor = F.pad(tensor, pad_width, mode='constant', value=pad_value)
        padded_tensors.append(padded_tensor)
    
    # Stack the padded tensors to create a batched tensor
    batched_tensor = torch.stack(padded_tensors)
    return batched_tensor


class SingleChainData(Dataset):
    def __init__(
        self,
        chain_dir=None,
        pickled_dir=None,
        force_process=True,
        limit_by=None,
        use_mask=True,
        use_crop=True,
        ignore_mask_fields=(),
        type_dict=None,
    ):
    
        self.chain_dir = chain_dir
        self.pickled_dir = pickled_dir
        self.limit_by = limit_by
        self.type_dict = type_dict

        if not os.path.exists(self.pickled_dir) or force_process:
            self.preprocess()

        self.length = len(os.listdir(self.pickled_dir))
        self.files = os.listdir(self.pickled_dir)
        self.use_mask = use_mask
        self.use_crop = use_crop
        self.ignore_mask_fields = ignore_mask_fields

    def preprocess(self):
        os.makedirs(self.pickled_dir, exist_ok=True)
        for i, file in enumerate(os.listdir(self.chain_dir)):
            if self.limit_by and i > self.limit_by:
                break
            if file.endswith(".ent"):
                try:
                    chain = TorchProtein.from_pdb(os.path.join(self.chain_dir, file), self.type_dict)
                    with open(
                        os.path.join(self.pickled_dir, file.replace(".ent", ".pkl")),
                        "wb",
                    ) as f:
                        pickle.dump(chain, f)
                except Exception as e:
                    print(f"Failed to load {file}")
                    print(e)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self.files[idx]
        with open(os.path.join(self.pickled_dir, f), "rb") as f:
            # trunk-ignore(bandit/B301)
            chain = pickle.load(f)

        data_fields = list(chain.__dataclass_fields__.keys())
        if self.use_crop:
            chain.crop_fields()
        if self.use_mask:
            mask = chain.mask_fields(ignore_mask_fields=self.ignore_mask_fields)

        geo_data = {}
        seq_data = {}
        raw_seq_data = {}
        for field in data_fields:
            field_data = getattr(chain, field)
            if isinstance(field_data, type(None)):
                continue
            if field_data.type_.pad_type == "torch_geometric":
                geo_data[field] = field_data.data
            elif field_data.type_.pad_type == "seq":
                seq_data[field] = field_data.data
            elif field_data.type_.pad_type == "esm":
                raw_seq_data[field] = field_data.data

        seq_data["mask"] = mask
        geo_data = Data.from_dict(geo_data)
        return geo_data, seq_data, raw_seq_data

class Collator:
    def __init__(self, type_dict, follow_key=None):
        self.follow_key = follow_key
        self.type_dict = type_dict

    def __call__(self, batch):
        geo_data_list = [d[0] for d in batch]
        seq_data_list = [d[1] for d in batch]
        # kind of gross this is a seperate edge case
        raw_seq_data_list = [
            (f"protein{i}", "".join(d[2]["sequence"])) for i, d in enumerate(batch)
        ]
        # check if any geometric data types
        if len(geo_data_list[0].keys()) == 0:
            batch_data = Batch()
        else:
            # eventually I'll need to change this to handle heterogenous data
            batch_data = Batch.from_data_list(geo_data_list, follow_batch=[self.follow_key])

            batch_data.batch = batch_data[f"{self.follow_key}_batch"]
            batch_data.ptr = batch_data[f"{self.follow_key}_ptr"]
            del batch_data[f"{self.follow_key}_batch"]
            del batch_data[f"{self.follow_key}_ptr"]

        for key in seq_data_list[0].keys():
            if key == "mask":
                list_of_tensors = [d[key] for d in seq_data_list]
                seq = pad_seqrep(list_of_tensors, key)
            elif self.type_dict[key].meta_data:
                seq = torch.tensor([d[key] for d in seq_data_list])
            elif not self.type_dict[key].pair_type:
                list_of_tensors = [d[key] for d in seq_data_list]
                seq = pad_seqrep(list_of_tensors, key)
            elif self.type_dict[key].pair_type:
                list_of_tensors = [d[key] for d in seq_data_list]
                seq = pad_pairrep(list_of_tensors)
                
            batch_data[key] = seq
        _, _, batch_tokens = batch_converter(raw_seq_data_list)
        batch_data.tokens = batch_tokens

        return batch_data
