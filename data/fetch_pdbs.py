import requests
import json
import os
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor
from utils import process_protein_inv

def fetch_pdb_ids(num_chains=1):
    url = 'https://search.rcsb.org/rcsbsearch/v2/query?json='
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "operator": "equals",
                "value": num_chains,
                "attribute": "rcsb_assembly_info.polymer_entity_instance_count"
            }
        },
        "request_options": {
            "return_all_hits": True
        },
        "return_type": "entry"
    }
    response = requests.post(url, json=query)
    if response.status_code == 200:
        results = response.json()
        pdb_ids = [entry['identifier'] for entry in results['result_set']]
        return pdb_ids
    else:
        print(f"Failed to fetch PDB IDs (HTTP {response.status_code})")
        print(response.text)
        print("Failed to fetch PDB IDs")
        return []
# Function to download PDB files based on a list of PDB IDs
def download_pdb_files(pdb_id):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir='pdbs', file_format='pdb', overwrite=False)

def download_inverse_folding_data(path_to_split_file, path_to_file):
    process_protein_inv(path_to_split_file, path_to_file)
    
    with open(os.path.join(path_to_file, "train.json")) as f:
        train = json.load(f)

    with open(os.path.join(path_to_file, "test.json")) as f:
        test = json.load(f)
    
    train_ids = list(train.keys())
    test_ids = list(test.keys())

    with ThreadPoolExecutor(max_workers=5) as executor:
        results_train = list(executor.map(download_pdb_files, train_ids))
        results_test = list(executor.map(download_pdb_files, test_ids))

# Main script execution
if __name__ == "__main__":
    download_inverse_folding_data("/Users/dylanabramson/Downloads/data/cath4.2/chain_set_splits.json", "./inverse_data")
