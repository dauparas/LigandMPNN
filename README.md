## LigandMPNN

This package provides inference code for [LigandMPNN](https://www.biorxiv.org/content/10.1101/2023.12.22.573103v1) & [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) models. The code and model parameters are available under MIT license.

### Dependencies
To run the model you will need to have Python>=3.0, PyTorch, Numpy installed, and to read/write PDB files you will need [Prody](https://pypi.org/project/ProDy/).

For example to make a new conda environment for LigandMPNN run:
```
conda create -n ligandmpnn_env python=3.11
pip3 install torch
pip install prody
```
### Model parameters
To download model parameters run:
```
bash get_model_params.sh "./model_params"
```

### Available models

To run the model of your choice run specify `--model_type` and optionally model checkpoint path. Available models:
- ProteinMPNN
```
--model_type "protein_mpnn"
--checkpoint_protein_mpnn "./model_params/proteinmpnn_v_48_002.pt" #noised with 0.02A Gaussian noise
--checkpoint_protein_mpnn "./model_params/proteinmpnn_v_48_010.pt" #noised with 0.10A Gaussian noise
--checkpoint_protein_mpnn "./model_params/proteinmpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
--checkpoint_protein_mpnn "./model_params/proteinmpnn_v_48_030.pt" #noised with 0.30A Gaussian noise
```
- LigandMPNN
```
--model_type "ligand_mpnn"
--checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_005_25.pt" #noised with 0.05A Gaussian noise
--checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" #noised with 0.10A Gaussian noise
--checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_020_25.pt" #noised with 0.20A Gaussian noise
--checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_030_25.pt" #noised with 0.30A Gaussian noise
```
- SolubleMPNN
```
--model_type "soluble_mpnn"
--checkpoint_soluble_mpnn "./model_params/solublempnn_v_48_002.pt" #noised with 0.02A Gaussian noise
--checkpoint_soluble_mpnn "./model_params/solublempnn_v_48_010.pt" #noised with 0.10A Gaussian noise
--checkpoint_soluble_mpnn "./model_params/solublempnn_v_48_020.pt" #noised with 0.20A Gaussian noise
--checkpoint_soluble_mpnn "./model_params/solublempnn_v_48_030.pt" #noised with 0.30A Gaussian noise
```
- ProteinMPNN with global membrane label
```
--model_type "global_label_membrane_mpnn"
--checkpoint_global_label_membrane_mpnn "./model_params/global_label_membrane_mpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
```
- ProteinMPNN with per residue membrane label
```
--model_type "per_residue_label_membrane_mpnn"
--checkpoint_per_residue_label_membrane_mpnn "./model_params/per_residue_label_membrane_mpnn_v_48_020.pt" #noised with 0.20A Gaussian noise
```

## Examples
### 1 default
Default setting will run ProteinMPNN.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/default"
```
### 2 --temperature
`--temperature 0.05` Change sampling temperature (higher temperature gives more sequence diversity).
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --temperature 0.05 \
        --out_folder "./outputs/temperature"
```
### 3 --seed
`--seed` Not selecting seed will run with a random seed. Running this multiple times will give different results.
```
python run.py \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/random_seed"
```
### 4 --verbose
`--verbose 0` Do not print any statements.
```
python run.py \
        --seed 111 \
        --verbose 0 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/verbose"
```
### 5 --save_stats
`--save_stats 1` Save sequence design statistics.
```
#['generated_sequences', 'sampling_probs', 'log_probs', 'decoding_order', 'native_sequence', 'mask', 'chain_mask', 'seed', 'temperature']
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/save_stats" \
        --save_stats 1
```
### 6 --fixed_residues
`--fixed_residues` Fixing specific amino acids. This examples fixes first 10 residues in chain C and adds global bias towards A (alanine). The output should have all alanines except the first 10 residues should be the same as in the input sequence since those are fixed.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/fix_residues" \
        --fixed_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        --bias_AA "A:10.0"
```

### 7 --redesigned_residues
`--redesigned_residues` Specifying which residues need to be designed. This example redesigns first 10 residues while fixing everything else.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/redesign_residues" \
        --redesigned_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        --bias_AA "A:10.0"
```

### 8 --number_of_batches
Design 15 sequences; with batch size 3 (can be 1 when using CPUs) and the number of batches 5.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/batch_size" \
        --batch_size 3 \
        --number_of_batches 5
```
### 9 --bias_AA
Global amino acid bias. In this example output sequences are biased towards W, P, C and away from A.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --bias_AA "W:3.0,P:3.0,C:3.0,A:-3.0" \
        --out_folder "./outputs/global_bias"
```
### 10 --bias_AA_per_residue
Specify per residue amino acid bias, e.g. make residues C1, C3, C5, C7 to be prolines.
```
# {
# "C1": {"G": -0.3, "C": -2.0, "P": 10.8},
# "C3": {"P": 10.0},
# "C5": {"G": -1.3, "P": 10.0},
# "C7": {"G": -1.3, "P": 10.0}
# }
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --bias_AA_per_residue "./inputs/bias_AA_per_residue.json" \
        --out_folder "./outputs/per_residue_bias"
```
### 11 --omit_AA
Global amino acid restrictions. This is equivalent to using `--bias_AA` and setting bias to be a large negative number. The output should be just made of E,K,A.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --omit_AA "CDFGHILMNPQRSTVWY" \
        --out_folder "./outputs/global_omit"
```

### 12 --omit_AA_per_residue
Per residue amino acid restrictions.
```
# {
# "C1": "ACDEFGHIKLMNPQRSTVW",
# "C3": "ACDEFGHIKLMNPQRSTVW",
# "C5": "ACDEFGHIKLMNPQRSTVW",
# "C7": "ACDEFGHIKLMNPQRSTVW"
# }
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --omit_AA_per_residue "./inputs/omit_AA_per_residue.json" \
        --out_folder "./outputs/per_residue_omit"
```
### 13 --symmetry_residues
### 13 --symmetry_weights
Designing sequences with symmetry, e.g. homooligomer/2-state proteins etc. In this example make C1=C2=C3, also C4=C5, and C6=C7.
```
#total_logits += symmetry_weights[t]*logits
#probs = torch.nn.functional.softmax((total_logits+bias_t) / temperature, dim=-1)
#total_logits_123 = 0.33*logits_1+0.33*logits_2+0.33*logits_3
#output should be ***ooxx
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/symmetry" \
        --symmetry_residues "C1,C2,C3|C4,C5|C6,C7" \
        --symmetry_weights "0.33,0.33,0.33|0.5,0.5|0.5,0.5"
```


### 14 --homo_oligomer
Design homooligomer sequences. This automatically sets `--symmetry_residues` and `--symmetry_weights` assuming equal weighting from all chains.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/homooligomer" \
        --homo_oligomer 1 \
        --number_of_batches 2
```

### 15 --file_ending
Outputs will have a specified ending; e.g. `1BC8_xyz.fa` instead of `1BC8.fa`
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/file_ending" \
        --file_ending "_xyz"
```

### 16 --zero_indexed
Zero indexed names in /backbones/1BC8_0.pdb, 1BC8_1.pdb, 1BC8_2.pdb etc
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/zero_indexed" \
        --zero_indexed 1 \
        --number_of_batches 2
```

### 17 --chains_to_design
Specify which chains (e.g. "ABC") need to be redesigned, other chains will be kept fixed. Outputs in seqs/backbones will still have atoms/sequence for the whole input PDB.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/chains_to_design" \
        --chains_to_design "B"
```
### 18 --parse_these_chains_only
Parse and design only specified chains (e.g. "ABC"). Outputs will have only specified chains.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/parse_these_chains_only" \
        --parse_these_chains_only "B"
```

### 19 --model_type "ligand_mpnn"
Run LigandMPNN with default settings.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_default"
```

### 20 --checkpoint_ligand_mpnn
Run LigandMPNN using 0.05A model by specifying `--checkpoint_ligand_mpnn` flag.
```
python run.py \
        --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_005_25.pt" \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_default"
```
### 21 --ligand_mpnn_use_atom_context
Setting `--ligand_mpnn_use_atom_context 0` will mask all ligand atoms. This can be used to access how much ligand atoms affect AA probabilities. 
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_no_context" \
        --ligand_mpnn_use_atom_context 0
```

### 22 --ligand_mpnn_use_side_chain_context
Use fixed residue side chain atoms as extra ligand atoms.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_use_side_chain_atoms" \
        --ligand_mpnn_use_side_chain_context 1 \
        --fixed_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"
```

### 23 --model_type "soluble_mpnn"
Run SolubleMPNN (ProteinMPNN like model with only soluble proteins in the training dataset).
```
python run.py \
        --model_type "soluble_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/soluble_mpnn_default"
```

### 24 --model_type "global_label_membrane_mpnn"
Run global label membrane MPNN (trained with extra input - binary label soluble vs not) `--global_transmembrane_label #1 - membrane, 0 - soluble`. 
```
python run.py \
        --model_type "global_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/global_label_membrane_mpnn_0" \
        --global_transmembrane_label 0
```

### 25 --model_type "per_residue_label_membrane_mpnn"
Run per residue label membrane MPNN (trained with extra input per residue specifying buried (hydrophobic), interface (polar), or other type residues; 3 classes).
```
python run.py \
        --model_type "per_residue_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/per_residue_label_membrane_mpnn_default" \
        --transmembrane_buried "C1 C2 C3 C11" \
        --transmembrane_interface "C4 C5 C6 C22"
```

### 26 --fasta_seq_separation
Choose a symbol to put between different chains in fasta output format. It's recommended to PDB output format to deal with residues jumps and multiple chain parsing.
```
python run.py \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/fasta_seq_separation" \
        --fasta_seq_separation ":"
```

### 27 --pdb_path_multi
Specify multiple PDB input paths. This is more efficient since the model needs to be loaded from the checkpoint once.
```
#{
#"./inputs/1BC8.pdb": "",
#"./inputs/4GYT.pdb": ""
#}
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --out_folder "./outputs/pdb_path_multi" \
        --seed 111
```

### 28 --fixed_residues_multi
Specify fixed residues when using `--pdb_path_multi` flag.
```
#{
#"./inputs/1BC8.pdb": "C1 C2 C3 C4 C5 C10 C22",
#"./inputs/4GYT.pdb": "A7 A8 A9 A10 A11 A12 A13 B38"
#}
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --fixed_residues_multi "./inputs/fix_residues_multi.json" \
        --out_folder "./outputs/fixed_residues_multi" \
        --seed 111
```

### 29 --redesigned_residues_multi
Specify which residues need to be redesigned when using `--pdb_path_multi` flag.
```
#{
#"./inputs/1BC8.pdb": "C1 C2 C3 C4 C5 C10",
#"./inputs/4GYT.pdb": "A7 A8 A9 A10 A12 A13 B38"
#}
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --redesigned_residues_multi "./inputs/redesigned_residues_multi.json" \
        --out_folder "./outputs/redesigned_residues_multi" \
        --seed 111
```

### 30 --omit_AA_per_residue_multi
Specify which residues need to be ommitted when using `--pdb_path_multi` flag.
```
#{
#"./inputs/1BC8.pdb": {"C1":"ACDEFGHILMNPQRSTVWY", "C2":"ACDEFGHILMNPQRSTVWY", "C3":"ACDEFGHILMNPQRSTVWY"},
#"./inputs/4GYT.pdb": {"A7":"ACDEFGHILMNPQRSTVWY", "A8":"ACDEFGHILMNPQRSTVWY"}
#}
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --omit_AA_per_residue_multi "./inputs/omit_AA_per_residue_multi.json" \
        --out_folder "./outputs/omit_AA_per_residue_multi" \
        --seed 111
```

### 31 --bias_AA_per_residue_multi
Specify amino acid biases per residue when using `--pdb_path_multi` flag.
```
#{
#"./inputs/1BC8.pdb": {"C1":{"A":3.0, "P":-2.0}, "C2":{"W":10.0, "G":-0.43}},
#"./inputs/4GYT.pdb": {"A7":{"Y":5.0, "S":-2.0}, "A8":{"M":3.9, "G":-0.43}}
#}
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --bias_AA_per_residue_multi "./inputs/bias_AA_per_residue_multi.json" \
        --out_folder "./outputs/bias_AA_per_residue_multi" \
        --seed 111
```

### 32 --ligand_mpnn_cutoff_for_score
This sets the cutoff distance in angstroms to select residues that are considered to be close to ligand atoms. This flag only affects the `num_ligand_res` and `ligand_confidence` in the output fasta files.
```
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --ligand_mpnn_cutoff_for_score "6.0" \
        --out_folder "./outputs/ligand_mpnn_cutoff_for_score"
```

### 33 specifying residues with insertion codes
You can specify residue using chain_id + residue_number + insersion_code; e.g. redesign only residue B82, B82A, B82B, B82C.
```
python run.py \
        --seed 111 \
        --pdb_path "./inputs/2GFB.pdb" \
        --out_folder "./outputs/insertion_code" \
        --redesigned_residues "B82 B82A B82B B82C" \
        --parse_these_chains_only "B"
```


### Citing this work
If you use the code, please cite:
```
@article{dauparas2023atomic,
  title={Atomic context-conditioned protein sequence design using LigandMPNN},
  author={Dauparas, Justas and Lee, Gyu Rie and Pecoraro, Robert and An, Linna and Anishchenko, Ivan and Glasscock, Cameron and Baker, David},
  journal={Biorxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}

@article{dauparas2022robust,
  title={Robust deep learning--based protein sequence design using ProteinMPNN},
  author={Dauparas, Justas and Anishchenko, Ivan and Bennett, Nathaniel and Bai, Hua and Ragotte, Robert J and Milles, Lukas F and Wicky, Basile IM and Courbet, Alexis and de Haas, Rob J and Bethel, Neville and others},
  journal={Science},
  volume={378},
  number={6615},  
  pages={49--56},
  year={2022},
  publisher={American Association for the Advancement of Science}
}
```