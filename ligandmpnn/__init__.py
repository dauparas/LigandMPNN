import json
import os
import random
import sys

import numpy as np
import torch

from prody import writePDB
from omegaconf import DictConfig

from ligandmpnn.data_utils import (
    alphabet,
    element_dict_rev,
    featurize,
    get_score,
    get_seq_rec,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
)
from ligandmpnn.model_utils import ProteinMPNN
from ligandmpnn.dataclass_utils import (
    MPNN_sequence,
    MPNN_WT_sequence,
    MPNN_Mutant_sequence,
    MPNN_weights,
)


# rewriten from run.py
class MPNN_designer:
    def __init__(self, config: DictConfig):
        self.cfg = config

        if self.cfg.runtime.mode.use not in self.cfg.runtime.mode.all:
            raise ValueError(f"Runtime mode must be one of {self.cfg.runtime.mode.all}")

        self.set_seed()
        self.set_device()
        self.setup_folders()
        self.set_model_parameters()
        self.load_model()
        self.load_pdb_paths()
        self.load_fixed_residues()
        self.load_redesigned_residues()

        if self.cfg.runtime.mode.use == "design":
            self.load_bias_information()
            self.load_omit_information()

    def set_seed(self):
        if not (seed := self.cfg.sampling.seed):
            seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])

        self.seed = seed

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        print(f"Seed: {self.seed}")

    def set_device(self):
        # self.device=device_picker.DevicePicker().pytorch_device(device='cpu' if self.cfg.runtime.force_cpu else '')
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and not self.cfg.runtime.force_cpu
            else "cpu"
        )
        print(f"Device:{self.device.type}: {self.device.index}")

    def setup_folders(self):
        self.base_folder = os.path.abspath(self.cfg.output.folder)
        self.seqs_folder = os.path.join(self.base_folder, "seqs")
        self.bb_folder = os.path.join(self.base_folder, "backbones")
        self.stats_folder = os.path.join(self.base_folder, "stats")

        for dir in (
            self.base_folder,
            self.seqs_folder,
            self.bb_folder,
            self.stats_folder,
        ):
            os.makedirs(dir, exist_ok=True)

    def set_model_parameters(self):
        model_types = {
            "protein_mpnn": self.cfg.checkpoint.protein_mpnn.use,
            "ligand_mpnn": self.cfg.checkpoint.ligand_mpnn.use,
            "per_residue_label_membrane_mpnn": self.cfg.checkpoint.per_residue_label_membrane_mpnn.use,
            "global_label_membrane_mpnn": self.cfg.checkpoint.global_label_membrane_mpnn.use,
            "soluble_mpnn": self.cfg.checkpoint.soluble_mpnn.use,
        }
        use_model = model_types.get(self.cfg.model_type.use)

        if not use_model:
            raise ValueError(
                "Invalid model_type provided. Please choose from: "
                + ", ".join(model_types.keys())
            )

        self.checkpoint_path = os.path.join(self.cfg.weight_dir, f"{use_model}.pt")

        os.makedirs(self.cfg.weight_dir, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            MPNN_weights().fetch_one_weights(
                download_dir=self.cfg.weight_dir, model=use_model
            )

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

    def load_pdb_paths(self):
        if pdb_path_multi := self.cfg.input.pdb_path_multi:
            with open(pdb_path_multi, "r") as fh:
                self.pdb_paths = list(json.load(fh))
        else:
            self.pdb_paths = [self.cfg.input.pdb]

    def load_fixed_residues(self):
        if fixed_residues_multi := self.cfg.input.fixed_residues_multi:
            with open(fixed_residues_multi, "r") as fh:
                self.fixed_residues_multi = json.load(fh)
        else:
            fixed_residues = [item for item in self.cfg.input.fixed_residues.split()]
            self.fixed_residues_multi = {pdb: fixed_residues for pdb in self.pdb_paths}

    def load_redesigned_residues(self):
        if redesigned_residues_multi := self.cfg.input.redesigned_residues_multi:
            with open(redesigned_residues_multi, "r") as fh:
                self.redesigned_residues_multi = json.load(fh)
        else:
            redesigned_residues = [
                item for item in self.cfg.input.redesigned_residues.split()
            ]
            self.redesigned_residues_multi = {
                pdb: redesigned_residues for pdb in self.pdb_paths
            }

    def load_bias_information(self):
        self.bias_AA = torch.zeros([21], device=self.device, dtype=torch.float32)
        if bias := self.cfg.input.bias.bias_AA:
            tmp = [item.split(":") for item in bias.split(",")]
            a1 = [b[0] for b in tmp]
            a2 = [float(b[1]) for b in tmp]
            for i, AA in enumerate(a1):
                self.bias_AA[restype_str_to_int[AA]] = a2[i]

        if bias_AA_per_residue_multi := self.cfg.input.bias.bias_AA_per_residue_multi:
            with open(bias_AA_per_residue_multi, "r") as fh:
                self.bias_AA_per_residue_multi = json.load(fh)
        else:
            self.bias_AA_per_residue_multi = {}
            if _b := self.cfg.input.bias.bias_AA_per_residue:
                with open(_b, "r") as fh:
                    bias_AA_per_residue = json.load(fh)
                for pdb in self.pdb_paths:
                    self.bias_AA_per_residue_multi[pdb] = bias_AA_per_residue

    def load_omit_information(self):
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
            if not residue_name in encoded_residues:
                continue
            i1 = encoded_residue_dict[residue_name]
            for amino_acid, v2 in v1.items():
                if not amino_acid in alphabet:
                    continue
                j1 = restype_str_to_int[amino_acid]
                bias_AA_per_residue[i1, j1] = v2

        return bias_AA_per_residue

    def get_omitted(self, pdb, encoded_residues, encoded_residue_dict):
        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self.device, dtype=torch.float32
        )
        if not (
            self.cfg.input.bias.omit_AA_per_residue_multi
            and self.cfg.input.bias.omit_AA_per_residue
        ):
            return omit_AA_per_residue

        omit_dict = self.omit_AA_per_residue_multi[pdb]
        for residue_name, v1 in omit_dict.items():
            if not residue_name in encoded_residues:
                continue
            i1 = encoded_residue_dict[residue_name]
            for amino_acid in v1:
                if not amino_acid in alphabet:
                    continue
                j1 = restype_str_to_int[amino_acid]
                omit_AA_per_residue[i1, j1] = 1.0

        return omit_AA_per_residue

    def get_feature_dict(self, protein_dict):
        # run featurize to remap R_idx and add batch dimension
        if self.cfg.runtime.verbose:
            if "Y" in list(protein_dict):
                atom_coords = protein_dict["Y"].cpu().numpy()
                atom_types = list(protein_dict["Y_t"].cpu().numpy())
                atom_mask = list(protein_dict["Y_m"].cpu().numpy())
                number_of_atoms_parsed = np.sum(atom_mask)
            else:
                print("No ligand atoms parsed")
                number_of_atoms_parsed = 0
                atom_types = ""
                atom_coords = []
            if number_of_atoms_parsed == 0:
                print("No ligand atoms parsed")
            elif self.cfg.model_type.use == "ligand_mpnn":
                print(
                    f"The number of ligand atoms parsed is equal to: {number_of_atoms_parsed}"
                )
                for i, atom_type in enumerate(atom_types):
                    print(
                        f"Type: {element_dict_rev[atom_type]}, Coords {atom_coords[i]}, Mask {atom_mask[i]}"
                    )
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=self.cfg.sampling.ligand_mpnn.cutoff_for_score,
            use_atom_context=self.cfg.sampling.ligand_mpnn.use_atom_context,
            number_of_ligand_atoms=self.atom_context_num,
            model_type=self.cfg.model_type.use,
        )
        feature_dict["batch_size"] = self.cfg.sampling.batch_size

        return feature_dict

    def linking_weights(
        self, encoded_residues, encoded_residue_dict, chain_letters_list
    ):
        # specify which residues are linked
        if symmetry_residues := self.cfg.input.symmetry.symmetry_residues:
            symmetry_residues_list_of_lists = [
                x.split(",") for x in symmetry_residues.split("|")
            ]
            remapped_symmetry_residues = []
            for t_list in symmetry_residues_list_of_lists:
                tmp_list = []
                for t in t_list:
                    tmp_list.append(encoded_residue_dict[t])
                remapped_symmetry_residues.append(tmp_list)
        else:
            remapped_symmetry_residues = [[]]

        # specify linking weights
        if self.cfg.input.symmetry.symmetry_weights:
            symmetry_weights = [
                [float(item) for item in x.split(",")]
                for x in self.cfg.input.symmetry.symmetry_weights.split("|")
            ]
        else:
            symmetry_weights = [[]]

        if self.cfg.input.symmetry.homo_oligomer:
            if self.cfg.runtime.verbose:
                print("Designing HOMO-OLIGOMER")
            chain_letters_set = list(set(chain_letters_list))
            reference_chain = chain_letters_set[0]
            lc = len(reference_chain)
            residue_indices = [
                item[lc:] for item in encoded_residues if item[:lc] == reference_chain
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

    def process_transmembrane(self, encoded_residues, fixed_positions, protein_dict):
        # specify which residues are buried for checkpoint_per_residue_label_membrane_mpnn model
        if buried := self.cfg.input.transmembrane.buried:
            buried_residues = [item for item in buried.split()]
            buried_positions = torch.tensor(
                [int(item in buried_residues) for item in encoded_residues],
                device=self.device,
            )
        else:
            buried_positions = torch.zeros_like(fixed_positions)

        if transmembrane_interface := self.cfg.input.transmembrane.interface:
            interface_residues = [item for item in transmembrane_interface.split()]
            interface_positions = torch.tensor(
                [int(item in interface_residues) for item in encoded_residues],
                device=self.device,
            )
        else:
            interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if self.cfg.model_type.use == "global_label_membrane_mpnn":
            protein_dict["membrane_per_residue_labels"] = (
                self.cfg.input.transmembrane.global_transmembrane_label
                + 0 * fixed_positions
            )

        return protein_dict

    def get_chain_mask(self, protein_dict):
        if isinstance(self.cfg.input.chains_to_design, str):
            chains_to_design_list = self.cfg.chains_to_design.split(",")
        else:
            chains_to_design_list = protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [
                    item in chains_to_design_list
                    for item in protein_dict["chain_letters"]
                ],
                dtype=np.int32,
            ),
            device=self.device,
        )

        return chain_mask

    def set_chain_mask(
        self,
        protein_dict,
        chain_mask,
        redesigned_residues,
        redesigned_positions,
        fixed_residues,
        fixed_positions,
        encoded_residue_dict_rev,
    ):
        # create chain_mask to notify which residues are fixed (0) and which need to be designed (1)
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        if self.cfg.runtime.verbose:
            PDB_residues_to_be_redesigned = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 1
            ]
            PDB_residues_to_be_fixed = [
                encoded_residue_dict_rev[item]
                for item in range(protein_dict["chain_mask"].shape[0])
                if protein_dict["chain_mask"][item] == 0
            ]
            print("These residues will be redesigned: ", PDB_residues_to_be_redesigned)
            print("These residues will be fixed: ", PDB_residues_to_be_fixed)

        return protein_dict

    def parse_protein(self, pdb):
        fixed_residues = self.fixed_residues_multi[pdb]
        redesigned_residues = self.redesigned_residues_multi[pdb]
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            pdb,
            device=self.device,
            chains=self.cfg.input.parse_these_chains_only,
            parse_all_atoms=self.cfg.sampling.ligand_mpnn.use_side_chain_context,
            parse_atoms_with_zero_occupancy=self.cfg.input.parse_atoms_with_zero_occupancy,
        )

        return (
            protein_dict,
            fixed_residues,
            redesigned_residues,
            other_atoms,
            backbone,
            icodes,
        )

    def get_encoded_residues(self, protein_dict, icodes):
        # make chain_letter + residue_idx + insertion_code mapping to integers
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())  # residue indices
        chain_letters_list = list(protein_dict["chain_letters"])  # chain letters
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)
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
        positions = torch.tensor(
            [int(item not in residues) for item in encoded_residues],
            device=self.device,
        )
        return positions

    def get_fixed(self, encoded_residues, fixed_residues):
        return self.get_positions(encoded_residues, fixed_residues)

    def get_redesigned(self, encoded_residues, redesigned_residues):
        return self.get_positions(encoded_residues, redesigned_residues)

    def run_sampling(self, feature_dict, name):
        sampling_probs_list = []
        log_probs_list = []
        decoding_order_list = []
        S_list = []
        loss_list = []
        loss_per_residue_list = []
        loss_XY_list = []
        for _ in range(self.cfg.sampling.number_of_batches):
            feature_dict["randn"] = torch.randn(
                [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                device=self.device,
            )
            output_dict = self.model.sample(feature_dict)

            # compute confidence scores
            loss, loss_per_residue = get_score(
                output_dict["S"],
                output_dict["log_probs"],
                feature_dict["mask"] * feature_dict["chain_mask"],
            )
            if self.cfg.model_type.use == "ligand_mpnn":
                combined_mask = (
                    feature_dict["mask"]
                    * feature_dict["mask_XY"]
                    * feature_dict["chain_mask"]
                )
            else:
                combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]
            loss_XY, _ = get_score(
                output_dict["S"], output_dict["log_probs"], combined_mask
            )
            # -----
            S_list.append(output_dict["S"])
            log_probs_list.append(output_dict["log_probs"])
            sampling_probs_list.append(output_dict["sampling_probs"])
            decoding_order_list.append(output_dict["decoding_order"])
            loss_list.append(loss)
            loss_per_residue_list.append(loss_per_residue)
            loss_XY_list.append(loss_XY)
        S_stack = torch.cat(S_list, 0)
        log_probs_stack = torch.cat(log_probs_list, 0)
        sampling_probs_stack = torch.cat(sampling_probs_list, 0)
        decoding_order_stack = torch.cat(decoding_order_list, 0)
        loss_stack = torch.cat(loss_list, 0)
        loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
        loss_XY_stack = torch.cat(loss_XY_list, 0)
        rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
        rec_stack = get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

        native_seq = "".join(
            [restype_int_to_str[AA] for AA in feature_dict["S"][0].cpu().numpy()]
        )

        out_dict = {}
        out_dict["generated_sequences"] = S_stack.cpu()
        out_dict["sampling_probs"] = sampling_probs_stack.cpu()
        out_dict["log_probs"] = log_probs_stack.cpu()
        out_dict["decoding_order"] = decoding_order_stack.cpu()
        out_dict["native_sequence"] = feature_dict["S"][0].cpu()
        out_dict["mask"] = feature_dict["mask"][0].cpu()
        out_dict["chain_mask"] = feature_dict["chain_mask"][0].cpu()
        out_dict["seed"] = self.seed
        out_dict["temperature"] = self.cfg.sampling.temperature

        if self.cfg.output.save_stats:
            output_stats_path = os.path.join(
                self.stats_folder, f"{name}{self.cfg.output.file_ending}.pt"
            )
            torch.save(out_dict, output_stats_path)

        return (
            rec_mask,
            rec_stack,
            combined_mask,
            S_stack,
            loss_stack,
            loss_XY_stack,
            loss_per_residue_stack,
            native_seq,
        )

    def design_proteins(self):
        # loop over PDB paths
        for pdb in self.pdb_paths:
            if self.cfg.runtime.verbose:
                print("Designing protein from this path:", pdb)

            (
                protein_dict,
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
            ) = self.get_encoded_residues(protein_dict, icodes)

            bias_AA_per_residue = self.get_biased(
                pdb, encoded_residues, encoded_residue_dict
            )
            omit_AA_per_residue = self.get_omitted(
                pdb, encoded_residues, encoded_residue_dict
            )

            fixed_positions = self.get_fixed(encoded_residues, fixed_residues)

            redesigned_positions = self.get_redesigned(
                encoded_residues, redesigned_residues
            )

            protein_dict = self.process_transmembrane(
                encoded_residues, fixed_positions, protein_dict
            )

            chain_mask = self.get_chain_mask(protein_dict)

            protein_dict = self.set_chain_mask(
                protein_dict,
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

            with torch.no_grad():
                feature_dict = self.get_feature_dict(protein_dict)

                B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                # add additional keys to the feature dictionary
                feature_dict["temperature"] = self.cfg.sampling.temperature
                feature_dict["bias"] = (
                    (-1e8 * self.omit_AA[None, None, :] + self.bias_AA).repeat(
                        [1, L, 1]
                    )
                    + bias_AA_per_residue[None]
                    - 1e8 * omit_AA_per_residue[None]
                )
                feature_dict["symmetry_residues"] = remapped_symmetry_residues
                feature_dict["symmetry_weights"] = symmetry_weights

                output_fasta = os.path.join(
                    self.seqs_folder, f"{name}{self.cfg.output.file_ending}.fa"
                )

                (
                    rec_mask,
                    rec_stack,
                    combined_mask,
                    S_stack,
                    loss_stack,
                    loss_XY_stack,
                    loss_per_residue_stack,
                    native_seq,
                ) = self.run_sampling(feature_dict, name)

                seq_np = np.array(list(native_seq))

                seq_out_str = []
                for mask in protein_dict["mask_c"]:
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
                    bool(self.cfg.sampling.ligand_mpnn.use_atom_context),
                    float(self.cfg.sampling.ligand_mpnn.cutoff_for_score),
                    self.cfg.sampling.batch_size,
                    self.cfg.sampling.number_of_batches,
                    self.checkpoint_path,
                )

                self.sequences.append(wt_seq)

                for ix in range(S_stack.shape[0]):
                    ix_suffix = ix
                    if not self.cfg.output.zero_indexed:
                        ix_suffix += 1
                    seq_rec_print = np.format_float_positional(
                        rec_stack[ix].cpu().numpy(), unique=False, precision=4
                    )
                    loss_np = np.format_float_positional(
                        np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
                    )
                    loss_XY_np = np.format_float_positional(
                        np.exp(-loss_XY_stack[ix].cpu().numpy()),
                        unique=False,
                        precision=4,
                    )
                    seq = "".join(
                        [restype_int_to_str[AA] for AA in S_stack[ix].cpu().numpy()]
                    )

                    # write new sequences into PDB with backbone coordinates
                    seq_prody = np.array([restype_1to3[AA] for AA in list(seq)])[
                        None,
                    ].repeat(4, 1)
                    bfactor_prody = (
                        loss_per_residue_stack[ix].cpu().numpy()[None, :].repeat(4, 1)
                    )
                    backbone.setResnames(seq_prody)
                    backbone.setBetas(
                        np.exp(-bfactor_prody)
                        * (bfactor_prody > 0.01).astype(np.float32)
                    )

                    writePDB(
                        filename=os.path.join(
                            self.bb_folder,
                            f"{name}_{str(ix_suffix)}{self.cfg.output.file_ending}.pdb",
                        ),
                        atoms=backbone + other_atoms if other_atoms else backbone,
                    )
                    # write fasta lines
                    seq_np = np.array(list(seq))
                    seq_out_str = []
                    for mask in protein_dict["mask_c"]:
                        seq_out_str += list(seq_np[mask.cpu().numpy()])
                        seq_out_str += [self.cfg.output.fasta_seq_separation]
                    seq_out_str = "".join(seq_out_str)[:-1]

                    variant: MPNN_Mutant_sequence = MPNN_Mutant_sequence(
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

                with open(output_fasta, "w") as handle:
                    for r in self.sequences:
                        handle.write(f">{r.id}\n{r.seq}\n")
                    handle.write("\n")

    # from score.py
    def score_proteins(self):
        # loop over PDB paths
        for pdb in self.pdb_paths:
            if self.cfg.runtime.verbose:
                print("Designing protein from this path:", pdb)

            (
                protein_dict,
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
            ) = self.get_encoded_residues(protein_dict, icodes)

            fixed_positions = self.get_fixed(encoded_residues, fixed_residues)

            redesigned_positions = self.get_redesigned(
                encoded_residues, redesigned_residues
            )

            protein_dict = self.process_transmembrane(
                encoded_residues, fixed_positions, protein_dict
            )

            chain_mask = self.get_chain_mask(protein_dict)

            protein_dict = self.set_chain_mask(
                protein_dict,
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

            with torch.no_grad():
                # run featurize to remap R_idx and add batch dimension
                feature_dict = self.get_feature_dict(protein_dict)

                B, L, _, _ = feature_dict["X"].shape  # batch size should be 1 for now.
                # add additional keys to the feature dictionary
                feature_dict["symmetry_residues"] = remapped_symmetry_residues

                logits_list = []
                probs_list = []
                log_probs_list = []
                decoding_order_list = []
                for _ in range(self.cfg.sampling.number_of_batches):
                    feature_dict["randn"] = torch.randn(
                        [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                        device=self.device,
                    )
                    if self.cfg.scorer.autoregressive_score:
                        score_dict = self.model.score(
                            feature_dict, use_sequence=self.cfg.scorer.use_sequence
                        )
                    elif self.cfg.scorer.single_aa_score:
                        score_dict = self.model.single_aa_score(
                            feature_dict, use_sequence=self.cfg.scorer.use_sequence
                        )
                    else:
                        print(
                            "Set either autoregressive_score or single_aa_score to True"
                        )
                        sys.exit()
                    logits_list.append(score_dict["logits"])
                    log_probs_list.append(score_dict["log_probs"])
                    probs_list.append(torch.exp(score_dict["log_probs"]))
                    decoding_order_list.append(score_dict["decoding_order"])
                log_probs_stack = torch.cat(log_probs_list, 0)
                logits_stack = torch.cat(logits_list, 0)
                probs_stack = torch.cat(probs_list, 0)
                decoding_order_stack = torch.cat(decoding_order_list, 0)

                output_stats_path = os.path.join(
                    self.stats_folder, f"{name}{self.cfg.output.file_ending}.pt"
                )
                out_dict = {}
                out_dict["logits"] = logits_stack.cpu().numpy()
                out_dict["probs"] = probs_stack.cpu().numpy()
                out_dict["log_probs"] = log_probs_stack.cpu().numpy()
                out_dict["decoding_order"] = decoding_order_stack.cpu().numpy()
                out_dict["native_sequence"] = feature_dict["S"][0].cpu().numpy()
                out_dict["mask"] = feature_dict["mask"][0].cpu().numpy()
                out_dict["chain_mask"] = (
                    feature_dict["chain_mask"][0].cpu().numpy()
                )  # this affects decoding order
                out_dict["seed"] = self.seed
                out_dict["alphabet"] = alphabet
                out_dict["residue_names"] = encoded_residue_dict_rev

                mean_probs = np.mean(out_dict["probs"], 0)
                std_probs = np.std(out_dict["probs"], 0)
                sequence = [
                    restype_int_to_str[AA] for AA in out_dict["native_sequence"]
                ]
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
                torch.save(out_dict, output_stats_path)
