#!/bin/bash
script_dir=$(dirname $0)
cd $script_dir
set -e 

#1 design a new sequence and pack side chains (return 1 side chain packing sample - fast) 
python scripts/run.py \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_default_fast" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=False \
        packer.pack_with_ligand_context=True

#2 design a new sequence and pack side chains (return 4 side chain packing samples) 
python scripts/run.py \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_default" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True


#3 fix specific residues for design and packing 
python scripts/run.py \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_fixed_residues" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True \
        input.fixed_residues="C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        packer.repack_everything=False

#4 fix specific residues for sequence design but repack everything 
python scripts/run.py \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_fixed_residues_full_repack" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=True \
        input.fixed_residues="C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        packer.repack_everything=True


#5 design a new sequence using LigandMPNN but pack side chains without considering ligand/DNA etc atoms 
python scripts/run.py \
        model_type.use="ligand_mpnn" \
        sampling.seed=111 \
        input.pdb="./inputs/1BC8.pdb" \
        output.folder="./outputs/sc_no_context" \
        packer.pack_side_chains=True \
        packer.number_of_packs_per_design=4 \
        packer.pack_with_ligand_context=False
