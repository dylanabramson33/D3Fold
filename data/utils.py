import json
import os


def process_protein_inv(path_to_split_file, path_to_file):
    """
    process the protein inverse folding dataset and save pdb id and chain pairs to a file
    Args:
        path_to_split_file: str, path to file containg train/val/test splits
    """
    with open(
        path_to_split_file,
    ) as f:
        split = json.load(f)
    train = split["train"]
    test = split["test"]
    train_data = [x.split(".") for x in train]
    test_data = [x.split(".") for x in test]

    # make new json mapping pdb id to chain
    pdb_chain_map_train = {}
    pdb_chain_map_test = {}

    os.makedirs(path_to_file, exist_ok=True)

    for pdb, chain in train_data:
        if pdb not in pdb_chain_map_train:
            pdb_chain_map_train[pdb] = chain

    for pdb, chain in test_data:
        if pdb not in pdb_chain_map_test:
            pdb_chain_map_test[pdb] = chain

    with open(os.path.join(path_to_file, "train.json"), "w") as f:
        json.dump(pdb_chain_map_train, f)

    with open(os.path.join(path_to_file, "test.json"), "w") as f:
        json.dump(pdb_chain_map_test, f)



