#!/bin/bash

#1
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/default"
#2
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --temperature 0.05 \
        --out_folder "./outputs/temperature"

#3
python run.py \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/random_seed"

#4
python run.py \
        --seed 111 \
        --verbose 0 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/verbose"

#5
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/save_stats" \
        --save_stats 1

#6
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/fix_residues" \
        --fixed_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        --bias_AA "A:10.0"

#7
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/redesign_residues" \
        --redesigned_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        --bias_AA "A:10.0"

#8
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/batch_size" \
        --batch_size 3 \
        --number_of_batches 5

#9
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --bias_AA "W:3.0,P:3.0,C:3.0,A:-3.0" \
        --out_folder "./outputs/global_bias"

#10
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --bias_AA_per_residue "./inputs/bias_AA_per_residue.json" \
        --out_folder "./outputs/per_residue_bias"

#11
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --omit_AA "CDFGHILMNPQRSTVWY" \
        --out_folder "./outputs/global_omit"

#12
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --omit_AA_per_residue "./inputs/omit_AA_per_residue.json" \
        --out_folder "./outputs/per_residue_omit"

#13
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/symmetry" \
        --symmetry_residues "C1,C2,C3|C4,C5|C6,C7" \
        --symmetry_weights "0.33,0.33,0.33|0.5,0.5|0.5,0.5"

#14
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/homooligomer" \
        --homo_oligomer 1 \
        --number_of_batches 2

#15
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/file_ending" \
        --file_ending "_xyz"

#16
python run.py \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/zero_indexed" \
        --zero_indexed 1 \
        --number_of_batches 2

#17
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/chains_to_design" \
        --chains_to_design "A,B"

#18
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/4GYT.pdb" \
        --out_folder "./outputs/parse_these_chains_only" \
        --parse_these_chains_only "A,B"

#19
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_default"

#20
python run.py \
        --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_005_25.pt" \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_v_32_005_25"

#21
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_no_context" \
        --ligand_mpnn_use_atom_context 0

#22
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/ligandmpnn_use_side_chain_atoms" \
        --ligand_mpnn_use_side_chain_context 1 \
        --fixed_residues "C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"

#23
python run.py \
        --model_type "soluble_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/soluble_mpnn_default"

#24
python run.py \
        --model_type "global_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/global_label_membrane_mpnn_0" \
        --global_transmembrane_label 0

#25
python run.py \
        --model_type "per_residue_label_membrane_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/per_residue_label_membrane_mpnn_default" \
        --transmembrane_buried "C1 C2 C3 C11" \
        --transmembrane_interface "C4 C5 C6 C22"

#26
python run.py \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/fasta_seq_separation" \
        --fasta_seq_separation ":"

#27
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --out_folder "./outputs/pdb_path_multi" \
        --seed 111

#28
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --fixed_residues_multi "./inputs/fix_residues_multi.json" \
        --out_folder "./outputs/fixed_residues_multi" \
        --seed 111

#29
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --redesigned_residues_multi "./inputs/redesigned_residues_multi.json" \
        --out_folder "./outputs/redesigned_residues_multi" \
        --seed 111

#30
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --omit_AA_per_residue_multi "./inputs/omit_AA_per_residue_multi.json" \
        --out_folder "./outputs/omit_AA_per_residue_multi" \
        --seed 111

#31
python run.py \
        --pdb_path_multi "./inputs/pdb_ids.json" \
        --bias_AA_per_residue_multi "./inputs/bias_AA_per_residue_multi.json" \
        --out_folder "./outputs/bias_AA_per_residue_multi" \
        --seed 111

#32
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --ligand_mpnn_cutoff_for_score "6.0" \
        --out_folder "./outputs/ligand_mpnn_cutoff_for_score"

#33
python run.py \
        --seed 111 \
        --pdb_path "./inputs/2GFB.pdb" \
        --out_folder "./outputs/insertion_code" \
        --redesigned_residues "B82 B82A B82B B82C" \
        --parse_these_chains_only "B"
