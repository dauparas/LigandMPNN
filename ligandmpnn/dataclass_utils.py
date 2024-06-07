from dataclasses import dataclass, field
import dataclasses
from typing import Any, Dict, List, Literal, Tuple, Union
from immutabledict import immutabledict
from abc import ABC, abstractmethod

import numpy as np
import torch

MPNN_MODEL_TYPE_HINT = Literal[
    "protein_mpnn",
    "soluble_mpnn",
    "ligand_mpnn",
    "per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn",
]
MPNN_MODEL_TYPE = (
    "protein_mpnn",
    "soluble_mpnn",
    "ligand_mpnn",
    "per_residue_label_membrane_mpnn",
    "global_label_membrane_mpnn",
)


@dataclass
class MPNN_Feature:
    model_type: MPNN_MODEL_TYPE_HINT

    # others
    randn: torch.Tensor = None

    # features
    mask_XY: torch.Tensor = None
    side_chain_mask: torch.Tensor = None
    Y: torch.Tensor = None
    Y_t: torch.Tensor = None
    Y_m: torch.Tensor = None
    membrane_per_residue_labels: torch.Tensor = None

    R_idx: torch.Tensor = None
    R_idx_original: torch.Tensor = None
    chain_labels: torch.Tensor = None
    S: torch.Tensor = None
    chain_mask: torch.Tensor = None
    mask: torch.Tensor = None
    X: torch.Tensor = None
    xyz_37: torch.Tensor = None
    xyz_37_m: torch.Tensor = None

    # configurations
    batch_size: int = None
    temperature: float = None

    # biases
    bias: torch.Tensor = None

    # fixed
    fixed_positions: torch.Tensor = None

    # symmetric
    symmetry_residues = None
    symmetry_weights = None

    # sc_features
    b_factors = None
    mean = None
    concentration = None
    mix_logits = None
    log_prob = None
    sample = None
    true_torsion_sin_cos = None
    X_m = None

    h_V: torch.Tensor = None
    h_E: torch.Tensor = None
    E_idx: torch.Tensor = None


@dataclass(frozen=True)
class MPNN_design:
    """
    Represents the design information of a protein, including its name, index, and relevant tensor dictionaries.
    These tensors are key to the message passing neural network (MPNN) model and its optimization process.
    """

    name: str
    idx: int
    tensor_dicts: Dict[str, Union[torch.Tensor, Any]]
    loss: torch.Tensor
    loss_per_residue: torch.Tensor

    loss_XY: torch.Tensor

    # fixed
    fixed_positions: torch.Tensor

    @property
    def S(self) -> torch.Tensor:
        """
        Gets the tensor S from the tensor dictionary. This tensor is generated sequence of the protein.

        Returns:
            torch.Tensor: The tensor S.
        """
        return self.tensor_dicts.get("S",None)

    @property
    def Sequences(self) -> Tuple[str]:
        """
        Converts tensors in the class attribute S to string sequences and returns a tuple of these sequences.

        This method processes the internally stored tensor data, transforming it into string sequences using a specific conversion function.

        Returns:
            A tuple of string sequences resulting from the conversion of tensors in S.
        """
        from .data_utils import tensor_to_sequence

        return tuple(tensor_to_sequence(tensor=t) for t in self.S)

    @property
    def log_probs(self) -> torch.Tensor:
        """
        Gets the log probabilities tensor from the tensor dictionary. This tensor may be used for calculating probabilities or losses in the model.

        Returns:
            torch.Tensor: The log probabilities tensor.
        """
        return self.tensor_dicts.get("log_probs")

    @property
    def sampling_probs(self) -> torch.Tensor:
        """
        Gets the sampling probabilities tensor from the tensor dictionary. This tensor is likely used for sampling operations during model training or inference.

        Returns:
            torch.Tensor: The sampling probabilities tensor.
        """
        return self.tensor_dicts.get("sampling_probs")
    
    @property
    def probs(self) -> torch.Tensor:
        """
        Gets the exp probabilities tensor from the tensor dictionary. This tensor is likely used for sampling operations during model training or inference.

        Returns:
            torch.Tensor: The sampling probabilities tensor.
        """
        return torch.exp(self.tensor_dicts.get("log_probs"))
    

    @property
    def decoding_order(self) -> torch.Tensor:
        """
        Gets the decoding order tensor from the tensor dictionary. This tensor may be used to specify the order of operations during the decoding process of the model.

        Returns:
            torch.Tensor: The decoding order tensor.
        """
        return self.tensor_dicts.get("decoding_order")
    

    @property
    def logits(self) -> torch.Tensor:
        """
        Gets the logits tensor from the tensor dictionary. This tensor may be used for calculating probabilities or losses in the model.

        Returns:
            torch.Tensor: The logits tensor.
        """
        return self.tensor_dicts.get("logits")
    
@dataclass
class MPNN_designs:
    """
    MPNN_designs class defines a class for molecular designs.
    This class serves for molecular design handling, providing basic design storage and identifier retrieval.
    Subclasses must implement the abstract method `id` to provide a unique identifier specific to the molecular design.
    """

    name: str
    model_type: MPNN_MODEL_TYPE_HINT
    designs: List[MPNN_design] = field(default_factory=list)
    native_seq: int = None

    @property
    def S_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.S for d in self.designs)

    @property
    def log_probs_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.log_probs for d in self.designs)

    @property
    def logits_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.logits for d in self.designs)

    @property
    def probs_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.probs for d in self.designs)

    @property
    def sampling_probs_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.sampling_probs for d in self.designs)

    @property
    def decoding_order_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.decoding_order for d in self.designs)

    @property
    def loss_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.loss for d in self.designs)

    @property
    def loss_per_residue_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.loss_per_residue for d in self.designs)

    @property
    def loss_XY_tuple(self) -> Tuple[torch.Tensor]:
        return tuple(d.loss_XY for d in self.designs)

    @property
    def S_stack(self) -> torch.Tensor:
        return torch.cat(self.S_tuple, 0)

    @property
    def log_probs_stack(self) -> torch.Tensor:
        return torch.cat(self.log_probs_tuple, 0)
    
    @property
    def logits_stack(self) -> torch.Tensor:
        return torch.cat(self.logits_tuple, 0)

    @property
    def probs_stack(self) -> torch.Tensor:
        return torch.cat(self.probs_tuple, 0)

    @property
    def sampling_probs_stack(self) -> torch.Tensor:
        return torch.cat(self.sampling_probs_tuple, 0)

    @property
    def decoding_order_stack(self) -> torch.Tensor:
        return torch.cat(self.decoding_order_tuple, 0)

    @property
    def loss_stack(self) -> torch.Tensor:
        return torch.cat(self.loss_tuple, 0)

    @property
    def loss_per_residue_stack(self) -> torch.Tensor:
        return torch.cat(self.loss_per_residue_tuple, 0)

    @property
    def loss_XY_stack(self) -> torch.Tensor:
        return torch.cat(self.loss_XY_tuple, 0)


    def __len__(self) -> int:
        return len(self.designs)

    def _get_seq(self)-> Dict[str, np.array]:
        return {'seq': np.array([s for d in self.designs for s in d.Sequences]) }

    def _get_loss(self)-> Dict[str, np.array]:
        return {'score': np.array([l for d in self.designs for l in d.loss_XY]) }

@dataclass
class MPNN_sequence(ABC):
    """
    MPNN_sequence class defines an abstract base class for molecular sequences.

    This class serves as a parent class for molecular sequence handling, providing basic sequence storage and identifier retrieval.
    Subclasses must implement the abstract method `id` to provide a unique identifier specific to the molecular sequence.

    Args:
    - seq: str, represents the molecular sequence as a string.
    """

    seq: str

    @property
    @abstractmethod
    def id(self):
        """
        Retrieves the unique identifier of the molecular sequence.

        This method is an abstract method that should be implemented in subclasses. It aims to provide a unique identifier for each molecular sequence.
        """
        ...


@dataclass
class MPNN_WT_sequence(MPNN_sequence):
    """
    Defines a subclass of Message Passing Neural Network (MPNN) for wild-type (WT) sequences, extending the base MPNN_sequence class.

    Attributes:
    - name: Name of the sequence.
    - T: Temperature parameter used in simulations.
    - seed: Random seed for reproducibility.
    - num_res: Number of residues in the structure.
    - num_ligand_res: Number of residues in the ligand.
    - use_ligand_context: Boolean indicating whether to incorporate ligand context.
    - ligand_cutoff_distance: Distance cutoff for considering ligand interactions.
    - batch_size: Size of data batches for training or inference.
    - number_of_batches: Total number of batches to process.
    - model_path: Path to the saved model.

    Property:
    - id: Generates a unique identifier string for the instance based on its attributes.
    """

    name: str
    T: float
    seed: int
    num_res: int
    num_ligand_res: int
    use_ligand_context: bool
    ligand_cutoff_distance: float
    batch_size: int
    number_of_batches: int
    model_path: str

    @property
    def id(self):
        """
        Returns a unique identifier string for the MPNN_WT_sequence instance, summarizing its key attributes.
        """
        return (
            f"{self.name}, T={self.T}, seed={self.seed}, num_res={self.num_res}, num_ligand_res={self.num_ligand_res}, "
            f"use_ligand_context={self.use_ligand_context}, ligand_cutoff_distance={self.ligand_cutoff_distance}, "
            f"batch_size={self.batch_size}, number_of_batches={self.number_of_batches}, model_path={self.model_path}"
        )


@dataclass
class MPNN_Mutant_sequence(MPNN_sequence):
    """
    The MPNN_Mutant_sequence class represents a specific type of molecule with mutated sequences.
    It inherits from the MPNN_sequence class and adds descriptions for properties particular to mutated sequences.

    Attributes:
    - name: The name of the molecule.
    - order: The ordinal number of the molecule.
    - T: The temperature parameter associated with the molecule.
    - seed: The random seed used for reproducibility.
    - overall_confidence: Confidence level in the overall structure prediction.
    - ligand_confidence: Confidence level specifically in the ligand binding prediction.
    - seq_rec: Sequence recovery score indicating how well the sequence is reconstructed.

    Properties:
    - id: A formatted string summarizing key attributes of the molecule, including name, order, temperature (T),
         seed, overall confidence, ligand confidence, and sequence recovery score.
    """

    name: str
    order: int
    T: float
    seed: int
    overall_confidence: float
    ligand_confidence: float
    seq_rec: float

    @property
    def id(self):
        """
        Generates a unique identifier string for the molecule based on its attributes.

        Returns:
        A formatted string containing the molecule's name, order, temperature (T), seed,
        overall confidence, ligand confidence, and sequence recovery score.
        """
        return (
            f"{self.name}, id={self.order}, T={self.T}, seed={self.seed}, "
            f"overall_confidence={self.overall_confidence}, ligand_confidence={self.ligand_confidence}, seq_rec={self.seq_rec}"
        )


@dataclass(frozen=True)
class MPNN_weights:
    model2md5: immutabledict = immutabledict(
        {
            "global_label_membrane_mpnn_v_48_020": "f95d52fe593e387ac852ce42a64ace7e",
            "ligandmpnn_v_32_005_25": "61cec2c47e680619e246694881d39af7",
            "ligandmpnn_v_32_010_25": "5cb0c0454d01132a463b576dc8ca43e5",
            "ligandmpnn_v_32_020_25": "c2488988cddcda60ce57d491d265084b",
            "ligandmpnn_v_32_030_25": "a93eedb3f9de277ab65f8013979dfcdf",
            "per_residue_label_membrane_mpnn_v_48_020": "cf4b842f446f399349064e1f6768073b",
            "proteinmpnn_v_48_002": "03e9ff81f6691580854123b9cb74efdf",
            "proteinmpnn_v_48_010": "4255760493a761d2b6cb0671a48e49b7",
            "proteinmpnn_v_48_020": "91d54c97a68bf551114f8c74c785e90f",
            "proteinmpnn_v_48_030": "b158144dd607a9662859f8fdc4add09f",
            "solublempnn_v_48_002": "9926175521f6539530ce6fa88fd7cc66",
            "solublempnn_v_48_010": "f5ad1edd1ffe094fc33d8c9b268d00c4",
            "solublempnn_v_48_020": "698982b1bda2b0d42e26538e64c93fda",
            "solublempnn_v_48_030": "e89591deba35ac4563d7da6e498d01e4",
            "ligandmpnn_sc_v_32_002_16": "9c5c2b71e8d449522ba8d9332a47b7bf",
        }
    )
    base_url = "https://files.ipd.uw.edu/pub/ligandmpnn/"

    def fetch_all_weights(self, download_dir):
        for m in self.model2md5.keys():
            self.fetch_one_weights(download_dir=download_dir, model=m)

    def fetch_one_weights(self, download_dir, model):
        if not (md5 := self.model2md5.get(model)):
            raise ValueError(f"invalid {model=}")

        import pooch

        pooch.retrieve(
            url=f"{self.base_url}/{model}.pt",
            known_hash=f"md5:{md5}",
            fname=f"{model}.pt",
            path=download_dir,
            progressbar=True,
        )
