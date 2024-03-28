#!/bin/bash

#make new directory for model parameters
#e.g.   bash get_model_params.sh "./model_params"

mkdir -p $1

#Original ProteinMPNN weights
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt -O $1"/proteinmpnn_v_48_002.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt -O $1"/proteinmpnn_v_48_010.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt -O $1"/proteinmpnn_v_48_020.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt -O $1"/proteinmpnn_v_48_030.pt"

#ProteinMPNN with num_edges=32
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_002.pt -O $1"/proteinmpnn_v_32_002.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_010.pt -O $1"/proteinmpnn_v_32_010.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_020.pt -O $1"/proteinmpnn_v_32_020.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_32_030.pt -O $1"/proteinmpnn_v_32_030.pt"

#LigandMPNN with num_edges=32; atom_context_num=25
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt -O $1"/ligandmpnn_v_32_005_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt -O $1"/ligandmpnn_v_32_010_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt -O $1"/ligandmpnn_v_32_020_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt -O $1"/ligandmpnn_v_32_030_25.pt"

#LigandMPNN with num_edges=32; atom_context_num=16
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_16.pt -O $1"/ligandmpnn_v_32_005_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_16.pt -O $1"/ligandmpnn_v_32_010_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_16.pt -O $1"/ligandmpnn_v_32_020_16.pt"
# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_16.pt -O $1"/ligandmpnn_v_32_030_16.pt"

# wget -q https://files.ipd.uw.edu/pub/ligandmpnn/publication_version_ligandmpnn_v_32_010_25.pt -O $1"/publication_version_ligandmpnn_v_32_010_25.pt"

#Per residue label membrane ProteinMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt -O $1"/per_residue_label_membrane_mpnn_v_48_020.pt"

#Global label membrane ProteinMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt -O $1"/global_label_membrane_mpnn_v_48_020.pt"

#SolubleMPNN
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt -O $1"/solublempnn_v_48_002.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt -O $1"/solublempnn_v_48_010.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt -O $1"/solublempnn_v_48_020.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt -O $1"/solublempnn_v_48_030.pt"

#LigandMPNN for side-chain packing (multi-step denoising model)
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt -O $1"/ligandmpnn_sc_v_32_002_16.pt"
