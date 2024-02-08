import requests
from Bio.PDB import PDBList
from concurrent.futures import ThreadPoolExecutor

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

# Main script execution
if __name__ == "__main__":
    resolution_threshold = 8.0  # Define your resolution threshold here
    pdb_ids = fetch_pdb_ids()
    with ThreadPoolExecutor(max_workers=5) as executor:
        # Use executor.map to apply the function to all items in parallel
        results = list(executor.map(download_pdb_files, pdb_ids))
