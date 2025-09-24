from enum import StrEnum


class Source(StrEnum):
    xyme = "xyme"
    washingtonUniversity = "washingtonUniversity"


parameter_file_uris = [
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_002.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_010.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_030.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_005_25.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_020_25.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_030_25.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/per_residue_label_membrane_mpnn_v_48_020.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/global_label_membrane_mpnn_v_48_020.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_002.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_010.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_030.pt",
    "https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt",
]
