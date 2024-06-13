from lightning.pytorch.loggers import WandbLogger
from lightning import Trainer

from D3Fold.data.protein import build_types
from D3Fold.model.folding_model import D3Fold
from D3Fold.data.torch_data import SingleChainData
from D3Fold.data.torch_data import Collator
from torch.utils.data import DataLoader

def main():
    type_dict = build_types()
    model = D3Fold()

    collator = Collator(type_dict)
    dataset = SingleChainData(
        chain_dir="./pdbs",
        pickled_dir="./pickled",
        use_mask=True,
        force_process=False,
        limit_by=None,
        type_dict=type_dict
    )
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collator)

    logger = WandbLogger(log_model="all")
    trainer = Trainer(logger=logger)
    trainer.fit(model=model, train_dataloaders=data_loader)



if __name__ == "__main__":
    main()
