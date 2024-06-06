#!/bin/bash
script_dir=$(dirname $0)
cd $script_dir

set -e 
#1
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/default"
#2
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        sampling.temperature=0.05 \
        output.folder="./outputs/temperature"

#3
ligandmpnn \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/random_seed"

#4
ligandmpnn \
        sampling.seed=111 \
        runtime.verbose=False \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/verbose"

#5
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/save_stats" \
        output.save_stats=True

#6
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/fix_residues" \
        input.fixed_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        input.bias.bias_AA="A:10.0"

#7
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/redesign_residues" \
        input.redesigned_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10" \
        input.bias.bias_AA="A:10.0"

#8
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/batch_size" \
        sampling.batch_size=3 \
        sampling.number_of_batches=5

#9
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.bias_AA=\"W:3.0,P:3.0,C:3.0,A:-3.0\" \
        output.folder="./outputs/global_bias"

#10
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.bias_AA_per_residue="./inputs/bias_AA_per_residue.json" \
        output.folder="./outputs/per_residue_bias"

#11
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.omit_AA="CDFGHILMNPQRSTVWY" \
        output.folder="./outputs/global_omit"

#12
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        input.bias.omit_AA_per_residue="./inputs/omit_AA_per_residue.json" \
        output.folder="./outputs/per_residue_omit"

#13
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/symmetry" \
        input.symmetry.symmetry_residues=\"C1,C2,C3+C4,C5+C6,C7\" \
        input.symmetry.symmetry_weights=\"0.33,0.33,0.33+0.5,0.5+0.5,0.5\"

#14
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/homooligomer" \
        input.symmetry.homo_oligomer=True \
        sampling.number_of_batches=2

#15
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/file_ending" \
        output.file_ending="_xyz"

#16
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/zero_indexed" \
        output.zero_indexed=True \
        sampling.number_of_batches=2

#17
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/chains_to_design" \
        input.chains_to_design=[A,B]

#18
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/4GYT.pdb" \
        output.folder="./outputs/parse_these_chains_only" \
        input.parse_these_chains_only=[A,B]

#19
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_default"

#20
ligandmpnn \
        checkpoint.ligand_mpnn.use="ligandmpnn_v_32_005_25" \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_v_32_005_25"

#21
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_no_context" \
        sampling.ligand_mpnn.use_atom_context=False 

#22
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/ligandmpnn_use_side_chain_atoms" \
        sampling.ligand_mpnn.use_side_chain_context=True \
        input.fixed_residues="C1 C2 C3 C4 C5 C6 C7 C8 C9 C10"

#23
ligandmpnn \
        model_type.use="soluble_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/soluble_mpnn_default"

#24
ligandmpnn \
        model_type.use="global_label_membrane_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/global_label_membrane_mpnn_0" \
        input.transmembrane.global_transmembrane_label=False 

#25
ligandmpnn \
        model_type.use="per_residue_label_membrane_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/per_residue_label_membrane_mpnn_default" \
        input.transmembrane.buried="C1 C2 C3 C11" \
        input.transmembrane.interface="C4 C5 C6 C22"

#26
ligandmpnn \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/fasta_seq_separation" \
        output.fasta_seq_separation=":"

#27
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        output.folder="./outputs/pdb_path_multi" \
        sampling.seed=111

#28
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.fixed_residues_multi="./inputs/fix_residues_multi.json" \
        output.folder="./outputs/fixed_residues_multi" \
        sampling.seed=111

#29
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.redesigned_residues_multi="./inputs/redesigned_residues_multi.json" \
        output.folder="./outputs/redesigned_residues_multi" \
        sampling.seed=111

#30
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.bias.omit_AA_per_residue_multi="./inputs/omit_AA_per_residue_multi.json" \
        output.folder="./outputs/omit_AA_per_residue_multi" \
        sampling.seed=111

#31
ligandmpnn \
        input.pdb_path_multi="./inputs/pdb_ids.json" \
        input.bias.bias_AA_per_residue_multi="./inputs/bias_AA_per_residue_multi.json" \
        output.folder="./outputs/bias_AA_per_residue_multi" \
        sampling.seed=111

#32
ligandmpnn \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        sampling.ligand_mpnn.cutoff_for_score="6.0" \
        output.folder="./outputs/ligand_mpnn_cutoff_for_score"

#33
ligandmpnn \
        sampling.seed=111 \
        input.pdb="./inputs/2GFB.pdb" \
        output.folder="./outputs/insertion_code" \
        input.redesigned_residues="B82 B82A B82B B82C" \
        input.parse_these_chains_only="B"

#34
mkdir -p customized_weight_dir_local
curl 'https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt' -o customized_weight_dir_local/customized_proteinmpnn_v_48_002.pt
ls customized_weight_dir_local
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_local" \
        checkpoint.customized.file=customized_weight_dir_local/customized_proteinmpnn_v_48_002.pt \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/default_customozed_weight_local"


#35
mkdir -p customized_weight_dir_remote
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_remote" \
        checkpoint.customized.url='https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt' \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/customized_weight_dir_remote"

ls customized_weight_dir_remote


#36
mkdir -p customized_weight_dir_remote_hash
ligandmpnn \
        sampling.seed=111 \
        weight_dir="customized_weight_dir_remote_hash" \
        checkpoint.customized.url='https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt' \
        checkpoint.customized.known_hash='md5:4255760493a761d2b6cb0671a48e49b7' \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/customized_weight_dir_remote_hash"

ls customized_weight_dir_remote_hash