from dataclasses import dataclass
from immutabledict import immutabledict
from abc import ABC, abstractmethod


@dataclass
class MPNN_sequence(ABC):
    seq: str

    @property
    @abstractmethod
    def id(self):
        ...


@dataclass
class MPNN_WT_sequence(MPNN_sequence):
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
        return f"{self.name}, T={self.T}, seed={self.seed}, num_res={self.num_res}, num_ligand_res={self.num_ligand_res}, use_ligand_context={self.use_ligand_context}, ligand_cutoff_distance={self.ligand_cutoff_distance}, batch_size={self.batch_size}, number_of_batches={self.number_of_batches}, model_path={self.model_path}"


@dataclass
class MPNN_Mutant_sequence(MPNN_sequence):
    name: str
    order: int
    T: float
    seed: int
    overall_confidence: float
    ligand_confidence: float
    seq_rec: float

    @property
    def id(self):
        return f"{self.name}, id={self.order}, T={self.T}, seed={self.seed}, overall_confidence={self.overall_confidence}, ligand_confidence={self.ligand_confidence}, seq_rec={self.seq_rec}"


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
            "ligandmpnn_sc_v_32_002_16": "9c5c2b71e8d449522ba8d9332a47b7bf"
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
