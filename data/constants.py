ONE_TO_THREE = {'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 'F':'PHE', 'G':'GLY', 'H':'HIS',
                'I':'ILE', 'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 'P':'PRO', 'Q':'GLN',
                'R':'ARG', 'S':'SER', 'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR', 'X':'XAA'}

THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}
THREE_TO_IND = {v: i for i, v in enumerate(ONE_TO_THREE.values())}
