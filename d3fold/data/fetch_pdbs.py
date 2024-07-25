import requests
import json
import os
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor
from d3fold.data.utils import process_protein_inv

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
def download_pdb_file(pdb_id, chain_ids=None):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(pdb_id, pdir='pdbs', file_format='pdb', overwrite=False)

def download_pdb_files(pdb_ids, chain_ids=None, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(download_pdb_file, pdb_ids))

def download_inverse_folding_data(path_to_split_file, path_to_file):
    process_protein_inv(path_to_split_file, path_to_file)
    
    with open(os.path.join(path_to_file, "train.json")) as f:
        train = json.load(f)

    with open(os.path.join(path_to_file, "test.json")) as f:
        test = json.load(f)
    
    train_ids = list(train.keys())
    test_ids = list(test.keys())

    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(download_pdb_files, train_ids))
        list(executor.map(download_pdb_files, test_ids))

import requests
import random
import json
import math
def get_sequence_length(entity_id):
    base_url = "https://data.rcsb.org/graphql"
    
    # Split the entity_id into PDB ID and entity number
    pdb_id, entity_num = entity_id.split('_')
    
    query = """
    query($pdb_id: String!, $entity_id: String!) {
      polymer_entity(entry_id: $pdb_id, entity_id: $entity_id) {
        rcsb_id
        entity_poly {
          rcsb_sample_sequence_length
        }
      }
    }
    """
    
    variables = {
        "pdb_id": pdb_id,
        "entity_id": entity_num
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(base_url, json={"query": query, "variables": variables}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data.get('data') and data['data'].get('polymer_entity'):
            return data['data']['polymer_entity']['entity_poly']['rcsb_sample_sequence_length']
    
    return None

def sample_random_chain(size_threshold, tolerance=10, max_attempts=100):
    base_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {
                            "from": size_threshold - tolerance,
                            "to": size_threshold + tolerance
                        }
                    }
                }
            ]
        },
        "return_type": "polymer_entity",
        "request_options": {
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
            "scoring_strategy": "combined",
        }
    }
    headers = {"Content-Type": "application/json"}

    response = requests.post(base_url, data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        data = response.json()
        result_set = data.get("result_set", [])

        if result_set:
            # Sort the results based on how close they are to the size threshold
            print(result_set)
            for result in result_set:
                print(result['identifier'], get_sequence_length(result['identifier']))
            

    return None, None, None

# Main script execution
if __name__ == "__main__":
    sample_random_chain(200, tolerance=2, max_attempts=100)
