#1 design a new sequence and pack side chains (return 1 side chain packing sample - fast) 
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/sc_default_fast" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 0 \
        --pack_with_ligand_context 1

#2 design a new sequence and pack side chains (return 4 side chain packing samples) 
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/sc_default" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 4 \
        --pack_with_ligand_context 1


#3 fix specific residues for design and packing 
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/sc_fixed_residues" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 4 \
        --pack_with_ligand_context 1 \
        --fixed_residues "C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        --repack_everything 0

#4 fix specific residues for sequence design but repack everything 
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/sc_fixed_residues_full_repack" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 4 \
        --pack_with_ligand_context 1 \
        --fixed_residues "C6 C7 C8 C9 C10 C11 C12 C13 C14 C15" \
        --repack_everything 1


#5 design a new sequence using LigandMPNN but pack side chains without considering ligand/DNA etc atoms 
python run.py \
        --model_type "ligand_mpnn" \
        --seed 111 \
        --pdb_path "./inputs/1BC8.pdb" \
        --out_folder "./outputs/sc_no_context" \
        --pack_side_chains 1 \
        --number_of_packs_per_design 4 \
        --pack_with_ligand_context 0
