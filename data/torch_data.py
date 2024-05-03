import os

import esm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from D3Fold.data.protein import TorchProtein

import cPickle as pickle
import gc



# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
FLOAT_TYPES = [torch.float32, torch.float64]
INT_TYPES = [torch.int32, torch.int64]

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
    ):
    
        self.chain_dir = chain_dir
        self.pickled_dir = pickled_dir
        self.limit_by = limit_by

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
                    chain = TorchProtein.from_pdb(os.path.join(self.chain_dir, file))
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
        f = open(f, "rb")

        # disable garbage collector
        gc.disable()
        chain = pickle.load(f)
        # enable garbage collector again
        gc.enable()
        f.close()

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
    def __init__(self, follow_key=None):
        self.follow_key = follow_key

    def __call__(self, batch):
        geo_data_list = [d[0] for d in batch]
        seq_data_list = [d[1] for d in batch]
        raw_seq_data_list = [
            (f"protein{i}", "".join(d[2]["raw_seq"])) for i, d in enumerate(batch)
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
            if seq_data_list[0][key].shape == ():
                batch_data[key] = torch.tensor([d[key] for d in seq_data_list])
            elif "mask" in key:
                seq = pad_sequence(
                    [d[key] for d in seq_data_list], batch_first=True, padding_value=False
                )
                batch_data[key] = seq
            elif seq_data_list[0][key].dtype in FLOAT_TYPES:
                seq = pad_sequence(
                    [d[key] for d in seq_data_list], batch_first=True, padding_value=torch.nan
                )
                batch_data[key] = seq
            elif seq_data_list[0][key].dtype in INT_TYPES:
                seq = pad_sequence(
                    [d[key] for d in seq_data_list], batch_first=True, padding_value=-1
                )
                batch_data[key] = seq
        _, _, batch_tokens = batch_converter(raw_seq_data_list)
        batch_data.tokens = batch_tokens

        return batch_data
