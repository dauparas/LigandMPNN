import copy
import dataclasses
import json
import os
import random
import sys
from typing import Any, Iterable, List, Literal, Union

import hydra
import numpy as np
import torch
import omegaconf

from prody import writePDB
from omegaconf import DictConfig

from ligandmpnn.data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    inspect_tensors,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    sequence_to_tensor,
    tensor_to_sequence,
    write_full_PDB,
)
from ligandmpnn.model_utils import ProteinMPNN
from ligandmpnn.dataclass_utils import (
    MPNN_Feature,
    MPNN_design,
    MPNN_designs,
    MPNN_sequence,
    MPNN_WT_sequence,
    MPNN_Mutant_sequence,
    MPNN_weights,
)



config_dir = os.path.join(os.path.dirname(__file__), "config")


@hydra.main(config_path=config_dir, config_name="ligandmpnn", version_base=None)
def run(cfg: DictConfig) -> None:
    """
    Main function for executing the inference process based on the configuration.
    
    Parameters:
    cfg (DictConfig): The configuration object containing all necessary parameters for execution.
    """
    """
    Inference function
    """
    
    # Initialize the MPNN designer based on the configuration
    magician: MPNN_designer = MPNN_designer(cfg)
    
    # Determine the execution mode from the configuration
    mode : Literal["design", "score"] = cfg.runtime.mode.use
    
    # Execute the corresponding mode function based on the determined mode
    if mode == "design":
        print(f"Mode: {mode}")
        magician.design_proteins()
    else:
        print(f"Mode: {mode}")
        magician.score_proteins()


# rewriten from run.py
class MPNN_designer:
    """
    Initializes the model class.

    Args:
    - config: DictConfig, configuration for the model's runtime settings.
    """

    def __init__(self, config: DictConfig):
        self.cfg = config

        # Validate the runtime mode against available modes.
        if self.cfg.runtime.mode.use not in self.cfg.runtime.mode.all:
            raise ValueError(f"Runtime mode must be one of {self.cfg.runtime.mode.all}")

        # Set random seed, device, folders, model parameters, and load the model.
        self.set_seed()
        self.set_device()
        self.setup_folders()
        self.set_model_parameters()
        self.load_model()

        # Initialize protein and feature dictionaries.
        self.protein_dict: dict[str,Union[torch.Tensor,Any]] = None
        self.feature_dict: MPNN_Feature = None

        # Skip further processing if no input PDB paths are provided.
        if not (self.cfg.input.pdb_path_multi or self.cfg.input.pdb):
            return

        # Load PDB paths, fixed residues, and redesigned residues.
        self.load_pdb_paths()
        self.load_fixed_residues()
        self.load_redesigned_residues()

        # If in 'design' mode, load bias and omit information.
        if self.cfg.runtime.mode.use == "design":
            self.load_bias_information()
            self.load_omit_information()

        # Determine whether to parse all atoms based on configuration flags.
        self.parse_all_atoms_flag = self.cfg.sampling.ligand_mpnn.use_side_chain_context or (
            self.cfg.packer.pack_side_chains and not self.cfg.packer.repack_everything
        )

    def set_seed(self):
        """
        Sets the random seed to ensure reproducibility of experiments.
        
        If a seed is not specified in the configuration, a random seed value is generated from a random number generator.
        
        Parameters:
            No explicit parameters other than the instance's configuration which is accessed via 'self.cfg'.
        
        Returns:
            None. The function modifies the instance's 'seed' attribute and sets seeds for various random number generators.
        
        Important Steps:
            1. Checks if a seed is provided in the configuration.
            2. Generates a random seed if none is found.
            3. Assigns the seed to the instance's 'seed' attribute.
            4. Sets the seed for PyTorch, Python's built-in random module, and NumPy's random module.
            5. Prints the set seed for reference.
        """
        if not (seed := self.cfg.sampling.seed):
            seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

        self.seed = seed

        torch.manual_seed(self.seed)  # Set PyTorch's random seed
        random.seed(self.seed)       # Set Python's random seed
        np.random.seed(self.seed)    # Set NumPy's random seed
        print(f"Seed: {self.seed}")   # Output the set seed for logging purposes

    def set_device(self):
        """
        Sets the device (CPU or GPU) based on availability and configuration.
        
        If CUDA is available and the configuration doesn't enforce CPU usage, the device is set to CUDA. 
        Otherwise, it defaults to CPU. The function also prints the selected device type and index.
        """
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self.cfg.runtime.force_cpu
            else "cpu"
        )
        print(f"Device: {self.device.type}: {self.device.index}")

    def setup_folders(self):
        """
        Sets up and creates necessary folders based on configurations for organizing output data.

        Important steps summarized:
        1. Determines the absolute path for the base output folder.
        2. Constructs paths for subfolders: sequences, backbones, statistics, and packed files.
        3. Iterates through these folders and creates them using `os.makedirs` with `exist_ok=True` 
           to avoid raising exceptions if the directories already exist.
        """
        self.base_folder = os.path.abspath(self.cfg.output.folder)
        self.seqs_folder = os.path.join(self.base_folder, "seqs")
        self.bb_folder = os.path.join(self.base_folder, "backbones")
        self.stats_folder = os.path.join(self.base_folder, "stats")
        self.packed_folder = os.path.join(self.base_folder, "packed")

        # Creates the required directories if they do not exist.
        for dir_path in (self.base_folder, self.seqs_folder, self.bb_folder, self.stats_folder):
            os.makedirs(dir_path, exist_ok=True)

    def set_model_parameters(self):
        """
        Set model parameters based on the configuration.

        Raises:
            ValueError: If an invalid `model_type` is provided.
            FileNotFoundError: If a custom checkpoint file is specified but not found.
            ValueError: If a custom checkpoint URL is given but not a valid HTTP/HTTPS link.

        """
        model_types = {
            "protein_mpnn": self.cfg.checkpoint.protein_mpnn.use,
            "ligand_mpnn": self.cfg.checkpoint.ligand_mpnn.use,
            "per_residue_label_membrane_mpnn": self.cfg.checkpoint.per_residue_label_membrane_mpnn.use,
            "global_label_membrane_mpnn": self.cfg.checkpoint.global_label_membrane_mpnn.use,
            "soluble_mpnn": self.cfg.checkpoint.soluble_mpnn.use,
            "ligandmpnn_sc": self.cfg.checkpoint.ligandmpnn_sc.use,
        }
        use_model = model_types.get(self.cfg.model_type.use)

        if not use_model:
            raise ValueError(
                "Invalid model_type provided. Please choose from: "
                + ", ".join(model_types.keys())
            )

        # Check for a custom checkpoint file or URL
        if file_path := self.cfg.checkpoint.customized.file:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"A customized checkpoint file is given but not found: {file_path}")
            self.checkpoint_path = file_path
        elif url_path := self.cfg.checkpoint.customized.url:
            if not (url_path.startswith("http://") or url_path.startswith("https://")):
                raise ValueError(f"A customized checkpoint url is given but not a valid url: {url_path}")
            model_basename = os.path.basename(url_path)

            import pooch

            known_hash = self.cfg.checkpoint.customized.known_hash

            pooch.retrieve(
                url=url_path,
                known_hash=known_hash if known_hash else None,
                fname=model_basename,
                path=self.cfg.weight_dir,
                progressbar=True,
            )
            self.checkpoint_path = os.path.join(self.cfg.weight_dir, model_basename)
        else:
            self.checkpoint_path = os.path.join(self.cfg.weight_dir, f"{use_model}.pt")

            os.makedirs(self.cfg.weight_dir, exist_ok=True)
            if not os.path.exists(self.checkpoint_path):
                MPNN_weights().fetch_one_weights(
                    download_dir=self.cfg.weight_dir, model=use_model
                )

        # Load the checkpoint and set additional attributes
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if self.cfg.model_type.use == "ligand_mpnn":
            self.atom_context_num = self.checkpoint["atom_context_num"]
            self.ligand_mpnn_use_side_chain_context = (
                self.cfg.sampling.ligand_mpnn.use_side_chain_context
            )
            self.k_neighbors = self.checkpoint["num_edges"]
        else:
            self.atom_context_num = 1
            self.ligand_mpnn_use_side_chain_context = 0
            self.k_neighbors = self.checkpoint["num_edges"]

    def load_model(self):
        """
        Loads the model and its parameters.
        
        This method initializes a ProteinMPNN model with specified configurations for primary structure processing. 
        It then loads the saved state dictionary, moves the model to the designated device, and sets it to evaluation mode. 
        If side chain packing is not enabled in the configuration, it skips the secondary model loading; otherwise, 
        it initializes a Packer model for side chain prediction, loads its weights, and sets it up for evaluation as well.
        """
        self.model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=self.k_neighbors,
            device=self.device,
            atom_context_num=self.atom_context_num,
            model_type=self.cfg.model_type.use,
            ligand_mpnn_use_side_chain_context=self.ligand_mpnn_use_side_chain_context,
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Skip side chain model loading if not configured
        if not self.cfg.packer.pack_side_chains:
            self.model_sc = None
            return

        # Initialize and load the side chain Packer model if side chain packing is enabled
        from ligandmpnn.sc_utils import Packer
        
        self.model_sc = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=self.device,
            num_mix=3,
        )
        
        # Prepare and load the checkpoint for the side chain model
        self.checkpoint_path_sc = os.path.join(self.cfg.weight_dir, f"{self.cfg.checkpoint.ligandmpnn_sc.use}.pt")
        os.makedirs(self.cfg.weight_dir, exist_ok=True)
        
        # Fetch weights if not locally available
        if not os.path.exists(self.checkpoint_path_sc):
            MPNN_weights().fetch_one_weights(
                download_dir=self.cfg.weight_dir, model=self.cfg.checkpoint.ligandmpnn_sc.use
            )

        checkpoint_sc = torch.load(self.checkpoint_path_sc, map_location=self.device)
        self.model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
        self.model_sc.to(self.device)
        self.model_sc.eval()

    def load_pdb_paths(self):
        """
        Load PDB paths from configuration.

        This function retrieves the list of PDB file paths based on the settings in the configuration.
        If a multi-PDB path is specified in `self.cfg.input.pdb_path_multi`, it reads the paths
        from the file and assigns them to `self.pdb_paths`. Otherwise, it sets `self.pdb_paths` to
        the single PDB path specified in `self.cfg.input.pdb`.
        """
        if pdb_path_multi := self.cfg.input.pdb_path_multi:
            with open(pdb_path_multi, "r") as fh:
                self.pdb_paths = list(json.load(fh))
        else:
            self.pdb_paths = [self.cfg.input.pdb]

    def load_fixed_residues(self):
        """
        Load fixed residue information.
        
        This function is responsible for reading or parsing fixed residue information from the configuration. If the 
        configuration specifies a multi-file for fixed residues, it opens and loads the data using JSON. Otherwise, it 
        processes a space-separated string of fixed residues into a dictionary, associating each PDB path with the same 
        list of fixed residues.
        """
        if fixed_residues_multi := self.cfg.input.fixed_residues_multi:
            # Load fixed residues from a multi-file specified in the configuration.
            with open(fixed_residues_multi, "r") as fh:
                self.fixed_residues_multi = json.load(fh)
        else:
            # Process a space-separated string of fixed residues into a dictionary for each PDB.
            fixed_residues = [item for item in self.cfg.input.fixed_residues.split()]
            self.fixed_residues_multi = {pdb: fixed_residues for pdb in self.pdb_paths}

    def load_redesigned_residues(self):
        """
        Loads the redesigned residues information based on the configuration provided.
        If a multi-file is specified, it reads from that file. Otherwise, it processes
        a single string input and applies the same residue list to all PDB structures.

        The function updates `self.redesigned_residues_multi`, which stores the redesigned
        residues for each PDB structure either from a file or derived from a common list.

        - **Attributes Updated**:
            - `self.redesigned_residues_multi`: A dictionary mapping PDB IDs to their respective
              redesigned residue lists.

        - **Flow**:
            1. Checks if a multi-residue file is specified in the configuration.
            2. If so, opens and loads the JSON file into `self.redesigned_residues_multi`.
            3. If not, splits a string of residues into a list and associates this list with
               each PDB ID in `self.pdb_paths`.
        """
        if redesigned_residues_multi := self.cfg.input.redesigned_residues_multi:
            with open(redesigned_residues_multi, "r") as fh:
                self.redesigned_residues_multi = json.load(fh)
        else:
            # If no multi-file is provided, create a uniform list for all PDBs from the input string
            redesigned_residues = [item for item in self.cfg.input.redesigned_residues.split()]
            self.redesigned_residues_multi = {pdb: redesigned_residues for pdb in self.pdb_paths}

    def load_bias_information(self):
        """
        Loads bias information.
        
        This method initializes the bias vector and loads two types of residue-specific biases 
        from configuration settings into `self.bias_AA` and `self.bias_AA_per_residue_multi`.
        """
        # Initialize a zero-filled tensor for bias values of 21 amino acids on the specified device.
        self.bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        
        # Load bias_AA values if provided in the configuration.
        if bias := self.cfg.input.bias.bias_AA:
            # Parse the bias string to separate amino acid codes and their corresponding bias values.
            tmp = [item.split(":") for item in bias.split(",")]
            a1, a2 = ([b[0] for b in tmp], [float(b[1]) for b in tmp])
            # Assign parsed bias values to the corresponding indices in the bias_AA tensor.
            for i, AA in enumerate(a1):
                self.bias_AA[restype_str_to_int[AA]] = a2[i]

        # Load per-residue multi bias information from a JSON file if specified.
        if bias_AA_per_residue_multi := self.cfg.input.bias.bias_AA_per_residue_multi:
            with open(bias_AA_per_residue_multi, "r") as fh:
                self.bias_AA_per_residue_multi = json.load(fh)
        else:
            # If bias_AA_per_residue_multi is not provided, check for bias_AA_per_residue and replicate its content for each PDB.
            self.bias_AA_per_residue_multi = {}
            if _b := self.cfg.input.bias.bias_AA_per_residue:
                with open(_b, "r") as fh:
                    bias_AA_per_residue = json.load(fh)
                # Assign the loaded bias values to each PDB path in the list.
                for pdb in self.pdb_paths:
                    self.bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    def load_omit_information(self):
        """
        Load information about amino acids to omit.

        This function initializes the `omit_AA_list` attribute with the amino acids to omit, 
        defined in the configuration under `cfg.input.bias.omit_AA`. It also creates a 
        tensor `omit_AA` with boolean values indicating whether each amino acid in the alphabet 
        should be omitted.

        If `omit_AA_per_residue_multi` is specified in the configuration, it reads the JSON file 
        and assigns the content to `self.omit_AA_per_residue_multi`. Otherwise, if 
        `omit_AA_per_residue` is provided, it populates `self.omit_AA_per_residue_multi` 
        with residue-specific omission data for each PDB path in `self.pdb_paths`.

        Args:
            self: An instance of the class containing necessary attributes like `cfg`, `pdb_paths`, `device`.

        Returns:
            None
        """
        self.omit_AA_list = self.cfg.input.bias.omit_AA
        self.omit_AA = torch.tensor(
            np.array([AA in self.omit_AA_list for AA in alphabet]).astype(np.float32),
            device=self.device,
        )
        if omit_AA_per_residue_multi := self.cfg.input.bias.omit_AA_per_residue_multi:
            with open(omit_AA_per_residue_multi, "r") as fh:
                self.omit_AA_per_residue_multi = json.load(fh)
        else:
            self.omit_AA_per_residue_multi = {}
            if _o := self.cfg.input.bias.omit_AA_per_residue:
                with open(_o, "r") as fh:
                    omit_AA_per_residue = json.load(fh)
                for pdb in self.pdb_paths:
                    self.omit_AA_per_residue_multi[pdb] = omit_AA_per_residue

    # after loadings
    def get_biased(self, pdb, encoded_residues, encoded_residue_dict):
        """
        Given the pdb information, retrieves bias values for each residue.
        
        Initializes a tensor to hold bias values for each residue with 21 possible amino acids,
        then checks configuration settings to determine whether to apply bias. If bias application is not enabled,
        returns the initialized zero tensor. Otherwise, it iterates through a bias dictionary specific to the given pdb,
        updating the bias tensor with values corresponding to residue names and their associated amino acids found in both
        the encoded residues and the predefined alphabet. The function finally returns the computed bias tensor.
        
        Args:
            pdb (str): The identifier of the protein data bank structure.
            encoded_residues (list): A list of encoded residue names in the structure.
            encoded_residue_dict (dict): A mapping from residue names to their respective encoding indices.
            
        Returns:
            torch.Tensor: A tensor of shape [number_of_residues, 21] containing bias values for each residue and possible amino acid.
        """
        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )
        if not (
            self.cfg.input.bias.bias_AA_per_residue_multi
            and self.cfg.input.bias.bias_AA_per_residue
        ):
            return bias_AA_per_residue
        bias_dict = self.bias_AA_per_residue_multi[pdb]
        for residue_name, v1 in bias_dict.items():
            if residue_name not in encoded_residues:
                continue
            i1 = encoded_residue_dict[residue_name]
            for amino_acid, v2 in v1.items():
                if amino_acid not in alphabet:
                    continue
                j1 = restype_str_to_int[amino_acid]
                bias_AA_per_residue[i1, j1] = v2

        return bias_AA_per_residue

    def get_omitted(self, pdb, encoded_residues, encoded_residue_dict):
        """
        Generates a tensor indicating omitted amino acids for each residue based on the PDB code.
        
        Args:
            pdb (str): The Protein Data Bank (PDB) identifier.
            encoded_residues (list): A list of encoded residue names.
            encoded_residue_dict (dict): A dictionary mapping residue names to their encoded indices.
            
        Returns:
            torch.Tensor: A float32 tensor of shape [len(encoded_residues), 21], 
                          where each row corresponds to a residue and columns indicate the presence 
                          (1.0) or absence (0.0) of omission for each of the 20 standard amino acids 
                          plus a potential additional category, aligned with the given device.
                          
        Important Steps:
            1. Initializes a zero tensor to hold omission status for each residue and amino acid.
            2. Checks configuration to decide whether to apply omission rules; if not, returns the initialized tensor.
            3. Retrieves specific omission rules for the given PDB from a predefined dictionary.
            4. Iterates over residues and their associated omitted amino acids, marking the omission in the tensor 
               if the amino acid is part of the standard alphabet and is found in the encoded residues.
        """
        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )
        
        # Skip processing if both conditions for omitting amino acids are not met in the configuration.
        if not (
            self.cfg.input.bias.omit_AA_per_residue_multi
            and self.cfg.input.bias.omit_AA_per_residue
        ):
            return omit_AA_per_residue

        omit_dict = self.omit_AA_per_residue_multi[pdb]
        
        # Iterate through the omission dictionary, marking the omission tensor accordingly.
        for residue_name, v1 in omit_dict.items():
            if residue_name not in encoded_residues:
                continue
            i1 = encoded_residue_dict[residue_name]
            for amino_acid in v1:
                if amino_acid not in alphabet:
                    continue
                j1 = restype_str_to_int[amino_acid]
                omit_AA_per_residue[i1, j1] = 1.0

        return omit_AA_per_residue

    def get_feature_dict(self):
        """
        Retrieves a feature dictionary.

        Runs featurization to remap R_idx and adds a batch dimension. If verbose mode is enabled in the config,
        it prints ligand-related information. Then, calls the `featurize` function with specified parameters
        and returns the resulting feature dictionary with an added 'batch_size' key.

        Parameters:
        self (object): The object instance with protein_dict and configuration attributes.

        Returns:
        dict: A dictionary containing features extracted from the protein_dict, with an additional 'batch_size' key.

        Note:
        - If no ligand atoms are parsed, a warning message will be printed.
        - Atom details are printed if using the "ligand_mpnn" model type.
        """
        # Run featurize to remap R_idx and add batch dimension
        if self.cfg.runtime.verbose:
            if "Y" in list(self.protein_dict):
                atom_coords = self.protein_dict["Y"].cpu().numpy()
                atom_types = list(self.protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(self.protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
            else:
                print("No ligand atoms parsed")
                number_of_atoms_parsed = 0
                atom_types = []
                atom_coords = []

            if number_of_atoms_parsed == 0:
                print("No ligand atoms parsed")
            elif self.cfg.model_type.use == "ligand_mpnn":
                print(f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}")
                for i, atom_type in enumerate(atom_types):
                    print(f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}")

        feature_dict = featurize(
            self.protein_dict,
            cutoff_for_score=self.cfg.sampling.ligand_mpnn.cutoff_for_score,
            use_atom_context=self.cfg.sampling.ligand_mpnn.use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.cfg.model_type.use,
        )
        feature_dict.batch_size=self.cfg.sampling.batch_size

        return feature_dict

    def linking_weights(
        self, encoded_residues, encoded_residue_dict, chain_letters_list
    ):
        """
        Generates linking weights based on the input configuration.

        Args:
            encoded_residues (list): List of encoded residue names.
            encoded_residue_dict (dict): Dictionary mapping residue names to encoded values.
            chain_letters_list (list): List of chain letters.

        Returns:
            tuple: A tuple containing two lists:
                - remapped_symmetry_residues (list): List of lists with encoded symmetry residues.
                - symmetry_weights (list): List of lists with weights for symmetry residues.
        """

        # Specify which residues are linked
        if symmetry_residues := self.cfg.input.symmetry.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in symmetry_residues.split("+")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        # Specify linking weights
        if self.cfg.input.symmetry.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in self.cfg.input.symmetry.symmetry_weights.split("+")
            ]
        else:
            symmetry_weights = [[]]

        # Handle Homo-oligomer design
        if self.cfg.input.symmetry.homo_oligomer:
            if self.cfg.runtime.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:]
                for item in encoded_residues
                if item[:lc] == reference_chain
            ]
            remapped_symmetry_residues = []
            symmetry_weights = []
            for res in residue_indices:
                tmp_list = []
                tmp_w_list = []
                for chain in chain_letters_set:
                    name = chain + res
                    tmp_list.append(encoded_residue_dict[name])
                    tmp_w_list.append(1 / len(chain_letters_set))
                remapped_symmetry_residues.append(tmp_list)
                symmetry_weights.append(tmp_w_list)

        return remapped_symmetry_residues, symmetry_weights

    def process_transmembrane(self, encoded_residues, fixed_positions):
        """
        Processes the transmembrane properties of the protein by assigning labels to residues based on their burial status 
        and interface involvement, preparing the data for models like 'checkpoint_per_residue_label_membrane_mpnn' 
        or applying a global transmembrane label as per the 'global_label_membrane_mpnn' model configuration.

        Parameters:
        - encoded_residues (List[str]): A list of encoded residue identifiers.
        - fixed_positions (torch.Tensor): A tensor indicating fixed positions, likely used for shaping outputs.

        Updates:
        Modifies `self.protein_dict["membrane_per_residue_labels"]` with calculated labels reflecting the residue's 
        transmembrane characteristics (buried, interface, or a combination).

        Note: The function does not return any value but directly modifies the `self.protein_dict` attribute.
        """
        # Determine buried residues based on configuration
        if buried := self.cfg.input.transmembrane.buried:
            buried_residues = buried.split()
            buried_positions = torch.tensor(
                [int(res in buried_residues) for res in encoded_residues],
                device=self.device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        # Determine interface residues based on configuration
        if transmembrane_interface := self.cfg.input.transmembrane.interface:
            interface_residues = transmembrane_interface.split()
            interface_positions = torch.tensor(
                [int(res in interface_residues) for res in encoded_residues],
                device=self.device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)

        # Assign per-residue membrane labels combining buried and interface statuses
        self.protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if self.cfg.model_type.use == "global_label_membrane_mpnn":
            self.protein_dict["membrane_per_residue_labels"] = (
                self.cfg.input.transmembrane.global_transmembrane_label
                + 0 * fixed_positions
            )

        return

    def get_chain_mask(self):
        """
        Generates a chain mask tensor based on the configuration, indicating which chains are to be designed.

        If chains to design are specified in the configuration, the mask is generated accordingly;
        otherwise, it defaults to using all chains present in the protein dictionary.

        Parameters:
            No explicit parameters are taken apart from the instance variables accessed via 'self'.

        Returns:
            chain_mask (torch.Tensor): A boolean tensor where each element corresponds to a chain 
                                      in the protein, True if the chain is to be designed, False otherwise.
                                      The tensor is located on the device specified in the instance.

        Raises:
            TypeError: If the type of chains_to_design is neither None, a string, nor an iterable.
        """
        
        # Determine chains to design based on configuration or default to all chains.
        if not (chains_to_design := self.cfg.input.chains_to_design):
            chains_to_design_list = self.protein_dict["chain_letters"]
        elif isinstance(chains_to_design, str):
            chains_to_design_list = chains_to_design.split(",")
        elif isinstance(chains_to_design, Iterable):
            chains_to_design_list = list(chains_to_design)
        else:
            raise TypeError(f"Unknown chain_letters type: {type(chains_to_design)}")
        
        # Create a boolean array indicating if each chain should be designed, then convert to a torch tensor.
        chain_mask = torch.tensor(
            np.array(
                [item in chains_to_design_list for item in self.protein_dict["chain_letters"]],
                dtype=np.int32,
            ),
            device=self.device,
        )

        return chain_mask
    def set_chain_mask(
        self,
        chain_mask,
        redesigned_residues,
        redesigned_positions,
        fixed_residues,
        fixed_positions,
        encoded_residue_dict_rev,
    ):
        """
        Sets the chain mask to indicate which residues are fixed (0) and which need to be redesigned (1).

        Args:
            chain_mask (np.ndarray): The initial mask for the protein chain.
            redesigned_residues (bool): Flag indicating whether there are residues to be redesigned.
            redesigned_positions (np.ndarray): Positions of residues designated for redesign.
            fixed_residues (bool): Flag indicating whether there are fixed residues.
            fixed_positions (np.ndarray): Positions of fixed residues.
            encoded_residue_dict_rev (dict): Reverse mapping of encoded residue IDs to PDB residue names.

        Returns:
            None: This function modifies the 'chain_mask' within 'protein_dict' in-place and prints 
                information about the residues to be fixed or redesigned based on configuration.

        The function updates 'chain_mask' in 'protein_dict' according to the provided parameters, 
        reflecting the design intentions for the protein residues. If verbose mode is enabled, it also 
        prints out the specific residues that are marked for redesign or to remain fixed.
        """
        # Create chain mask based on redesign or fixed residue requirements
        if redesigned_residues:
            self.protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            self.protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            self.protein_dict["chain_mask"] = chain_mask

        # Optionally print verbose output detailing fixed and redesigned residues
        if self.cfg.runtime.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(self.protein_dict["chain_mask"].shape[0])
                if self.protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(self.protein_dict["chain_mask"].shape[0])
                if self.protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        return None

    def parse_protein(self, pdb):
        """
        Parses protein structure data.

        Extracts protein structure information from the given PDB file, including fixed residues, 
        redesigned residues, chain lists, and other atomic information. Utilizes predefined device 
        settings and parsing flags to process the PDB file and returns the relevant information.

        Args:
        pdb (str): The identifier of the Protein Data Bank (PDB) structure to be parsed.

        Returns:
        tuple: A tuple containing:
            - fixed_residues (dict): Residues that are fixed in the structure.
            - redesigned_residues (dict): Residues that have been redesigned.
            - other_atoms (list): List of additional atom coordinates and details.
            - backbone (list): Coordinates and properties of the protein backbone atoms.
            - icodes (list): List of insertion codes indicating variations at a given residue position.
        """
        fixed_residues = self.fixed_residues_multi[pdb]
        redesigned_residues = self.redesigned_residues_multi[pdb]
        chain_list = self.get_chain_list()

        # Parses the PDB file to extract structural information based on specified parameters.
        self.protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=self.device,
            chains=chain_list,
            parse_all_atoms=self.parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=self.cfg.input.parse_atoms_with_zero_occupancy,
        )

        return (
            fixed_residues,
            redesigned_residues,
            other_atoms,
            backbone,
            icodes,
        )
    
    def get_chain_list(self) -> List:
        """
        Retrieves a list of chains to be parsed based on the configuration.

        Returns an empty list if parsing is not restricted to specific chains.
        If 'parse_these_chains_only' is a string, it's split by commas into a list.
        If it's an iterable, it's converted to a list directly.
        Otherwise, returns an empty list.

        :return: List of chains to parse or an empty list if no specific chains are set.
        """
        parse_these_chains_only = self.cfg.input.parse_these_chains_only
        if not parse_these_chains_only:
            return []

        if isinstance(parse_these_chains_only, str) and "," in parse_these_chains_only:
            return parse_these_chains_only.split(",")
        
        if isinstance(parse_these_chains_only, Iterable):
            return list(parse_these_chains_only)
        
        return []

    def get_encoded_residues(self, icodes):
        """
        Encodes protein residues by combining chain letter, residue index, and insertion code into a string.
        Returns a list of encoded residues and dictionaries for mapping between the encoded strings and their indices.

        Args:
        icodes: A list containing insertion codes for each residue.

        Returns:
        - encoded_residues: A list of encoded residue identifiers.
        - encoded_residue_dict: A dictionary mapping encoded residues to their indices.
        - encoded_residue_dict_rev: A dictionary mapping indices to encoded residues.
        - chain_letters_list: A list of chain letters.
        """

        # Extract residue indices from the protein dictionary as a numpy array
        # Create a mapping from chain letter + residue index + insertion code to integers
        R_idx_list = list(self.protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(self.protein_dict["chain_letters"])  # chain letters

        # Encode residues by concatenating chain letter, residue index, and insertion code
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)

        # Build dictionaries for encoding and decoding residue indices
        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))
        encoded_residue_dict_rev = dict(
            zip(list(range(len(encoded_residues))), encoded_residues)
        )

        return (
            encoded_residues,
            encoded_residue_dict,
            encoded_residue_dict_rev,
            chain_letters_list,
        )

    def get_positions(self, encoded_residues, residues):
        """
        Generates a tensor indicating positions of encoded residues that are not present in the given residue set.

        Parameters:
        - encoded_residues (list): The list of encoded residues.
        - residues (set): The set of specific residues to compare against.

        Returns:
        - positions (torch.Tensor): A tensor where each element corresponds to an encoded residue, 
          with a value of 1 if the residue is not in the 'residues' set, and 0 otherwise. The tensor is on the same device as self.
        """
        positions = torch.tensor(
            [int(item not in residues) for item in encoded_residues],
            device=self.device,
        )
        return positions

    def get_fixed(self, encoded_residues, fixed_residues):
        return self.get_positions(encoded_residues, fixed_residues)

    def get_redesigned(self, encoded_residues, redesigned_residues):
        """
        Retrieves the redesigned residue positions.

        This method identifies the positions where residues have been redesigned by 
        comparing the encoded residue sequence with the redesigned residue sequence.

        Args:
            encoded_residues (list): A list representing the encoded amino acid residues.
            redesigned_residues (list): A list representing the redesigned amino acid residues.

        Returns:
            list: A list of positions where residues have been redesigned.
        """
        return self.get_positions(encoded_residues, redesigned_residues)

    def run_sampling(self, name):
        """
        Execute the sampling process.

        Parameters:
        name (str): The name identifier for the sampling process, used for naming output files.

        Returns:
        tuple: A tuple containing various results and statistics from the sampling process, including
        reconstruction mask, reconstruction sequence, combined mask, stacked prediction sequences,
        loss stack, loss on XY stack, loss per residue stack, native sequence, and list of prediction sequences.
        """
        # Initialize lists to store results from each sampling batch

        new_mpnn_designs=MPNN_designs(name=name,model_type=self.cfg.model_type.use)
        # Convert the native sequence to a string format
        native_seq = tensor_to_sequence(self.feature_dict.S[0])

        print(native_seq)

        new_mpnn_designs.native_seq=native_seq

        # Loop through the specified number of batches
        for _ in range(self.cfg.sampling.number_of_batches):
            # Generate random noise and update the feature dictionary
            self.feature_dict.randn = torch.randn(
                [self.feature_dict.batch_size, self.feature_dict.mask.shape[1]],
                device=self.device
            )

            # Call the model's sampling method to get the output dictionary
            output_dict = self.model.sample(self.feature_dict)
            
            # Calculate the loss and loss per residue for the current batch
            # compute confidence scores
            loss, loss_per_residue = get_score(
                output_dict["S"],
                output_dict["log_probs"],
                self.feature_dict.mask * self.feature_dict.chain_mask,
            )
            # Determine the combined mask based on the model type
            if self.cfg.model_type.use == "ligand_mpnn":
                combined_mask = (
                    self.feature_dict.mask
                    * self.feature_dict.mask_XY
                    * self.feature_dict.chain_mask
                )
            else:
                combined_mask = (
                    self.feature_dict.mask * self.feature_dict.chain_mask
                )
            # Calculate the loss on XY for the current batch
            loss_XY, _ = get_score(
                output_dict["S"], output_dict["log_probs"], combined_mask
            )
            output_dict['mask']=self.feature_dict.mask
            output_dict['combined_mask']=combined_mask

            new_design=MPNN_design(
                name=name, 
                idx=_,
                tensor_dicts=output_dict,
                loss=loss,
                loss_per_residue=loss_per_residue,
                loss_XY=loss_XY,
                fixed_positions=self.feature_dict.fixed_positions)
            
            print(new_design.Sequences)
            #inspect_tensors(out_dict=dataclasses.asdict(new_design))
            inspect_tensors(out_dict=new_design.tensor_dicts)

            # Store the results of the current batch in the respective lists
            new_mpnn_designs.designs.append(new_design)
            
        
        # Calculate the reconstruction sequence and mask
        rec_mask = self.feature_dict.mask[:1] * self.feature_dict.chain_mask[:1]
        rec_stack = get_seq_rec(self.feature_dict.S[:1], new_mpnn_designs.S_stack, rec_mask)


        # Prepare the output dictionary containing various results and statistics
        out_dict = {}
        out_dict["generated_sequences"] = new_mpnn_designs.S_stack.cpu()
        out_dict["sampling_probs"] = new_mpnn_designs.sampling_probs_stack.cpu()
        out_dict["log_probs"] = new_mpnn_designs.log_probs_stack.cpu()
        out_dict["decoding_order"] = new_mpnn_designs.decoding_order_stack.cpu()
        out_dict["native_sequence"] = self.feature_dict.S[0].cpu()
        out_dict["mask"] = self.feature_dict.mask[0].cpu()
        out_dict["chain_mask"] = self.feature_dict.chain_mask[0].cpu()
        out_dict["seed"] = self.seed
        out_dict["temperature"] = self.cfg.sampling.temperature

        #inspect_tensors(out_dict=out_dict)
        print(new_mpnn_designs._get_seq())
        print(new_mpnn_designs._get_loss())

        # If configured to save statistics, save the output dictionary to a file
        if self.cfg.output.save_stats:
            output_stats_path = os.path.join(
                self.stats_folder, f"{name}{self.cfg.output.file_ending}.pt"
            )
            torch.save(out_dict, output_stats_path)

        # Return various results and statistics from the sampling process
        return (
            rec_mask,
            rec_stack,
            combined_mask,
            new_mpnn_designs
        )
    
    def sampling_sc(self, S_list):
        """
        Samples side chain conformations using a Monte Carlo approach.

        Parameters:
        S_list: A list containing multiple side chain structures.

        Returns:
        A tuple containing results from multiple packing runs, including X_stack_list, X_m_stack_list, b_factor_stack_list.
        """

        # Imports the side chain packing utility if not already present
        from ligandmpnn.sc_utils import pack_side_chains

        # Prints progress information based on the configuration's verbosity setting
        if self.cfg.runtime.verbose:
            print("Packing side chains...")

        # Generates feature representations for the protein structure to prepare for side chain packing
        feature_dict_ = featurize(
            self.protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=self.cfg.packer.pack_with_ligand_context,
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )

        # Deep copies the feature dictionary to avoid modifying the original
        _sc_feature_dict =dataclasses.asdict(feature_dict_)

        # Configures batch size from the configuration
        B = self.cfg.sampling.batch_size

        # Repeats feature tensors to match the batch size for each key in the dictionary
        for k, v in _sc_feature_dict.items():
            if k != "S":
                try:
                    num_dim = len(v.shape)  # Determines the number of dimensions in the tensor
                    # Repeats tensor along the batch dimension accordingly
                    _sc_feature_dict[k] = [v.repeat(B, *([1]*(num_dim-1)))][0]
                except Exception:  # Passes silently if the operation fails for any reason
                    pass


        sc_feature_dict=MPNN_Feature(**_sc_feature_dict)

        # Initialize lists to hold packed side chain data across batches
        X_stack_list, X_m_stack_list, b_factor_stack_list = [], [], []

        # Iterates for the specified number of packing runs per design
        for _ in range(self.cfg.packer.number_of_packs_per_design):
            # Resets lists for each batch within a packing run
            X_list, X_m_list, b_factor_list = [], [], []

            # Processes each batch within the current packing run
            for c in range(self.cfg.sampling.number_of_batches):
                # Assigns the current side chain structure to the feature dictionary
                sc_feature_dict.S = S_list[c]

                # Packs side chains using the current feature dictionary and model settings
                sc_dict = pack_side_chains(
                    sc_feature_dict,
                    self.model_sc,
                    self.cfg.packer.sc_num_denoising_steps,
                    self.cfg.packer.sc_num_samples,
                    self.cfg.packer.repack_everything,
                )

                # Collects outputs from the side chain packing function
                X_list.append(sc_dict.X)
                X_m_list.append(sc_dict.X_m)
                b_factor_list.append(sc_dict.b_factors)

            # Concatenates collected data across batches into tensors
            X_stack = torch.cat(X_list, 0)
            X_m_stack = torch.cat(X_m_list, 0)
            b_factor_stack = torch.cat(b_factor_list, 0)

            # Appends stacked tensors to the respective lists for the current packing run
            X_stack_list.append(X_stack)
            X_m_stack_list.append(X_m_stack)
            b_factor_stack_list.append(b_factor_stack)
        
        # Returns the collected data from all packing runs
        return X_stack_list, X_m_stack_list, b_factor_stack_list

    def design_proteins(self):
        """
        Process protein design for each PDB file path in the list.
        
        This function iterates over the list of PDB file paths, prints the current processing path, and calls the `design_proteins_single` method to perform protein design on the corresponding PDB file.
        """
        # loop over PDB paths
        for pdb in self.pdb_paths:
            print(f"Processing {pdb=}")
            self.design_proteins_single(pdb=pdb)

    def design_proteins_single(self, pdb):
        """
        Designs a single protein from the given PDB file.

        This function parses the provided PDB, encodes it, and uses a deep learning model for protein design.
        It involves determining fixed and redesigned residue positions, processing transmembrane regions, and handling symmetry residues.
        Finally, it generates designed protein sequences and structure files.

        Args:
        - pdb (str): Path to the PDB file.
        """
        if self.cfg.runtime.verbose:
            print("Designing protein from this path:", pdb)

        # Parse the protein from the PDB file
        (fixed_residues, redesigned_residues, other_atoms, backbone, icodes) = self.parse_protein(pdb)

        # Encode the residues and obtain necessary dictionaries
        (encoded_residues, encoded_residue_dict, encoded_residue_dict_rev, chain_letters_list) = self.get_encoded_residues(icodes)

        # Calculate biased and omitted amino acids per residue
        bias_AA_per_residue = self.get_biased(pdb, encoded_residues, encoded_residue_dict)
        omit_AA_per_residue = self.get_omitted(pdb, encoded_residues, encoded_residue_dict)

        # Identify fixed and redesigned positions
        fixed_positions = self.get_fixed(encoded_residues, fixed_residues)
        redesigned_positions = self.get_redesigned(encoded_residues, redesigned_residues)

        # Process transmembrane region
        self.process_transmembrane(encoded_residues, fixed_positions)

        # Create chain mask
        chain_mask = self.get_chain_mask()

        # Set chain mask and prepare for sampling
        self.set_chain_mask(
            chain_mask,
            redesigned_residues,
            redesigned_positions,
            fixed_residues,
            fixed_positions,
            encoded_residue_dict_rev,
        )

        # Compute symmetry-related information
        remapped_symmetry_residues, symmetry_weights = self.linking_weights(encoded_residues, encoded_residue_dict, chain_letters_list)

        # Zero out other atom B-factors
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # Adjust PDB name and prepare for output
        name:str = os.path.basename(pdb)
        if name.endswith(".pdb"):
            name = name[:-4]

        # Perform sampling without gradient calculation
        with torch.no_grad():
            self.feature_dict = self.get_feature_dict()

            # Add additional features to the feature dictionary
            B, L, _, _ = self.feature_dict.X.shape  # Batch size should be 1 for now.
            self.feature_dict.temperature = self.cfg.sampling.temperature
            self.feature_dict.bias = (
                (-1e8 * self.omit_AA[None, None, :] + self.bias_AA).repeat([1, L, 1])
                + bias_AA_per_residue[None]
                - 1e8 * omit_AA_per_residue[None]
            )
            self.feature_dict.fixed_positions = fixed_positions
            self.feature_dict.symmetry_residues = remapped_symmetry_residues
            self.feature_dict.symmetry_weights = symmetry_weights

            # Run sampling and obtain results
            output_fasta = os.path.join(self.seqs_folder, f"{name}{self.cfg.output.file_ending}.fa")
            (rec_mask, rec_stack, combined_mask, mpnn_designs) = self.run_sampling(name)

            # If packing side chains, process and save packed structures
            if self.cfg.packer.pack_side_chains:
                X_stack_list, X_m_stack_list, b_factor_stack_list = self.sampling_sc(mpnn_designs.S_tuple)

            # Convert native sequence to numpy array and create output sequence string
            seq_np = np.array(list(mpnn_designs.native_seq))

            seq_out_str = []
            for mask in self.protein_dict["mask_c"]:
                seq_out_str += list(seq_np[mask.cpu().numpy()])
                seq_out_str += [self.cfg.output.fasta_seq_separation]

            seq_out_str = "".join(seq_out_str)[:-1]

            self.sequences: list[MPNN_sequence] = []

            wt_seq: MPNN_WT_sequence = MPNN_WT_sequence(
                seq_out_str,
                name,
                self.cfg.sampling.temperature,
                self.seed,
                torch.sum(rec_mask).cpu().numpy(),
                torch.sum(combined_mask[:1]).cpu().numpy(),
                self.cfg.sampling.ligand_mpnn.use_atom_context,
                self.cfg.sampling.ligand_mpnn.cutoff_for_score,
                self.cfg.sampling.batch_size,
                self.cfg.sampling.number_of_batches,
                self.checkpoint_path,
            )
            self.sequences.append(wt_seq)

            # Create mutant sequences and save designed PDB structures
            for ix in range(mpnn_designs.S_stack.shape[0]):
                ix_suffix = ix
                if not self.cfg.output.zero_indexed:
                    ix_suffix += 1
                seq_rec_print = np.format_float_positional(
                    rec_stack[ix].cpu().numpy(), unique=False, precision=4
                )
                loss_np = np.format_float_positional(
                    np.exp(-mpnn_designs.loss_stack[ix].cpu().numpy()), unique=False, precision=4
                )
                loss_XY_np = np.format_float_positional(
                    np.exp(-mpnn_designs.loss_XY_stack[ix].cpu().numpy()),
                    unique=False,
                    precision=4,
                )
                seq = tensor_to_sequence(tensor=mpnn_designs.S_stack[ix])

                if self.cfg.output.save_bb_pdb:
                    # write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                        None,
                    ].repeat(4, 1)
                    bfactor_prody = (
                        mpnn_designs.loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                    )
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(
                        np.exp(-bfactor_prody) * (bfactor_prody > 0.01).astype(np.float32)
                    )

                    writePDB(
                        os.path.join(
                            self.bb_folder,
                            f"{name}_{str(ix_suffix)}{self.cfg.output.file_ending}.pdb",
                        ),
                        backbone + other_atoms if other_atoms else backbone,
                    )

                # Write full PDB files if packing side chains
                if self.cfg.packer.pack_side_chains:
                    for c_pack in range(self.cfg.packer.number_of_packs_per_design):
                        X_stack:torch.Tensor = X_stack_list[c_pack]
                        X_m_stack:torch.Tensor = X_m_stack_list[c_pack]
                        b_factor_stack:torch.Tensor = b_factor_stack_list[c_pack]
                        filename = os.path.join(
                            self.packed_folder,
                            f'{self.cfg.packer.packed_suffix}_{str(ix_suffix)}_{str(c_pack + 1)}{self.cfg.output.file_ending}.pdb',
                        )
                        write_full_PDB(
                            filename,
                            X_stack[ix].cpu().numpy(),
                            X_m_stack[ix].cpu().numpy(),
                            b_factor_stack[ix].cpu().numpy(),
                            self.feature_dict.R_idx_original[0].cpu().numpy(),
                            self.protein_dict["chain_letters"],
                            mpnn_designs.S_stack[ix].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                            force_hetatm=self.cfg.packer.force_hetatm,
                        )

                if self.cfg.output.save_fasta:
                    # Create mutant sequence object and append to sequences list
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in self.protein_dict["mask_c"]:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += [self.cfg.output.fasta_seq_separation]
                    seq_out_str = "".join(seq_out_str)[:-1]

                    variant = MPNN_Mutant_sequence(
                        seq_out_str,
                        name,
                        ix_suffix,
                        self.cfg.sampling.temperature,
                        self.seed,
                        loss_np,
                        loss_XY_np,
                        seq_rec_print,
                    )
                    self.sequences.append(variant)

                # Write sequences to a FASTA file
                with open(output_fasta, "w") as handle:
                    for r in self.sequences:
                        handle.write(f">{r.id}\n{r.seq}\n")
                    handle.write("\n")

    def score_proteins(self):
        # loop over PDB paths
        for pdb in self.pdb_paths:
            print(f"Processing {pdb=}")
            self.score_proteins_single(pdb=pdb)

    # from score.py
    def score_proteins_single(self, pdb):
        """
        Scores a single protein structure.

        This function processes a single protein structure file, predicting scores for various design and optimization scenarios.
        It starts by parsing the PDB file, encoding the amino acid sequence, handling transmembrane regions, identifying fixed and redesign positions,
        processing symmetric residues, and finally running the model for scoring.

        Args:
            pdb (str): Path to the protein structure file in PDB format.

        Raises:
            RuntimeError: If neither `autoregressive_score` nor `single_aa_score` is set to True in the configuration.

        Important steps:
            1. Parse protein from the given PDB file.
            2. Encode amino acids and prepare residue dictionaries.
            3. Identify fixed and redesign positions.
            4. Process transmembrane regions.
            5. Handle chain mask and symmetry residues.
            6. Run the model for scoring using either autoregressive or single amino acid scoring method.
            7. Save output statistics to a `.pt` file.

        Note:
            The function assumes the batch size is 1 and adjusts the input PDB name by removing the ".pdb" extension if present.
        """
        if not (
            self.cfg.scorer.autoregressive_score or self.cfg.scorer.single_aa_score
        ):
            raise RuntimeError(
                "Set either autoregressive_score or single_aa_score to True"
            )

        if self.cfg.runtime.verbose:
            print("Designing protein from this path:", pdb)

        (
            fixed_residues,
            redesigned_residues,
            other_atoms,
            backbone,
            icodes,
        ) = self.parse_protein(pdb)

        (
            encoded_residues,
            encoded_residue_dict,
            encoded_residue_dict_rev,
            chain_letters_list,
        ) = self.get_encoded_residues(icodes)

        fixed_positions = self.get_fixed(encoded_residues, fixed_residues)

        redesigned_positions = self.get_redesigned(
            encoded_residues, redesigned_residues
        )

        self.process_transmembrane(encoded_residues, fixed_positions)

        chain_mask = self.get_chain_mask()

        self.set_chain_mask(
            chain_mask,
            redesigned_residues,
            redesigned_positions,
            fixed_residues,
            fixed_positions,
            encoded_residue_dict_rev,
        )

        remapped_symmetry_residues, symmetry_weights = self.linking_weights(
            encoded_residues, encoded_residue_dict, chain_letters_list
        )

        # set other atom bfactors to 0.0
        if other_atoms:
            other_bfactors = other_atoms.getBetas()
            other_atoms.setBetas(other_bfactors * 0.0)

        # adjust input PDB name by dropping .pdb if it does exist
        name = os.path.basename(pdb)

        if name.endswith(".pdb"):
            name = name[:-4]

        new_mpnn_designs=MPNN_designs(name=name,model_type=self.cfg.model_type.use)
        

        with torch.no_grad():
            # run featurize to remap R_idx and add batch dimension
            self.feature_dict = self.get_feature_dict()

            # Convert the native sequence to a string format
            native_seq = tensor_to_sequence(self.feature_dict.S[0])
            print(native_seq)
            new_mpnn_designs.native_seq=native_seq

            B, L, _, _ = self.feature_dict.X.shape  # batch size should be 1 for now.
            # add additional keys to the feature dictionary
            self.feature_dict.symmetry_residues = remapped_symmetry_residues

            customized_seq=self.cfg.scorer.customized_seq
            print(f'{len(customized_seq)=}')
            print(f'{self.feature_dict.S.shape}')

            if customized_seq != '' and customized_seq is not None:
                customized_sequence=sequence_to_tensor(self.cfg.scorer.customized_seq)
                print(f'{customized_sequence.shape=}')
                # Decide whether to use the predefined sequence or the custom sequence
            
                print(f'use customized sequence: {customized_sequence=}')
                self.feature_dict.S=customized_sequence
            else:
                customized_sequence=None
                self.feature_dict.S=self.feature_dict.S.to(torch.int64)
                print(f'use pdb sequence: {self.feature_dict.S=}')
                


            for _ in range(self.cfg.sampling.number_of_batches):
                self.feature_dict.randn = torch.randn(
                    [
                        self.feature_dict.batch_size,
                        self.feature_dict.mask.shape[1],
                    ],
                    device=self.device,
                )
            
                if self.cfg.scorer.autoregressive_score:
                    score_dict = self.model.score(
                        self.feature_dict, use_sequence=self.cfg.scorer.use_sequence
                    )
                else:
                    score_dict = self.model.single_aa_score(
                        self.feature_dict, use_sequence=self.cfg.scorer.use_sequence
                    )

                

                loss, loss_per_residue = get_score(
                    score_dict["S"],
                    score_dict["log_probs"],
                    self.feature_dict.mask * self.feature_dict.chain_mask,
                )

                # Determine the combined mask based on the model type
                if self.cfg.model_type.use == "ligand_mpnn":
                    combined_mask = (
                        self.feature_dict.mask
                        * self.feature_dict.mask_XY
                        * self.feature_dict.chain_mask
                    )
                else:
                    combined_mask = (
                        self.feature_dict.mask * self.feature_dict.chain_mask
                    )
                # Calculate the loss on XY for the current batch
                loss_XY, _ = get_score(
                    score_dict["S"], score_dict["log_probs"], combined_mask
                )
                score_dict['mask']=self.feature_dict.mask
                score_dict['combined_mask']=combined_mask


                new_design=MPNN_design(
                    name=name, 
                    idx=_,
                    tensor_dicts=score_dict,
                    loss=loss,
                    loss_per_residue=loss_per_residue,
                    loss_XY=loss_XY,
                    fixed_positions=self.feature_dict.fixed_positions)
                
                # Store the results of the current batch in the respective lists
                new_mpnn_designs.designs.append(new_design)

            log_probs_stack = new_mpnn_designs.log_probs_stack.cpu()
            logits_stack = new_mpnn_designs.logits_stack.cpu()
            probs_stack =new_mpnn_designs.probs_stack.cpu()
            decoding_order_stack = new_mpnn_designs.decoding_order_stack.cpu()

            
            out_dict = {}
            out_dict["logits"] = logits_stack.cpu().numpy()
            out_dict["probs"] = probs_stack.cpu().numpy()
            out_dict["log_probs"] = log_probs_stack.cpu().numpy()
            out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
            out_dict["native_sequence"] = self.feature_dict.S[0].cpu().numpy()
            out_dict["mask"] = self.feature_dict.mask[0].cpu().numpy()
            out_dict["chain_mask"] = (
                self.feature_dict.chain_mask[0].cpu().numpy()
            )  # this affects decoding order
            out_dict["seed"] = self.seed
            out_dict["alphabet"] = alphabet
            out_dict["residue_names"] = encoded_residue_dict_rev

            mean_probs = np.mean(out_dict["probs"], 0)
            std_probs = np.std(out_dict["probs"], 0)
            sequence = [restype_int_to_str[AA] for AA in out_dict["native_sequence"]]
            mean_dict = {}
            std_dict = {}
            for residue in range(L):
                mean_dict_ = dict(zip(alphabet, mean_probs[residue]))
                mean_dict[encoded_residue_dict_rev[residue]] = mean_dict_
                std_dict_ = dict(zip(alphabet, std_probs[residue]))
                std_dict[encoded_residue_dict_rev[residue]] = std_dict_

            out_dict["sequence"] = sequence
            out_dict["mean_of_probs"] = mean_dict
            out_dict["std_of_probs"] = std_dict

            if self.cfg.runtime.verbose:
                #inspect_tensors(out_dict)

                print(new_mpnn_designs._get_seq())
                print(new_mpnn_designs._get_loss())

            if self.cfg.output.save_stats:
                output_stats_path = os.path.join(
                self.stats_folder, f"{name}{self.cfg.output.file_ending}.pt"
            )
                torch.save(out_dict, output_stats_path)
