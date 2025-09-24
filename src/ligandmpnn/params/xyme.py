import multiprocessing
import os
from pathlib import Path

import boto3
from joblib import Parallel, delayed

from ligandmpnn.logger import logger
from ligandmpnn.params.common import parameter_file_uris
from ligandmpnn.paths import MODEL_PARAMS_DIR

LIGANDMPNN_MODEL_PARAMS_BUCKET = os.environ.get(
    "LIGANDMPNN_MODEL_PARAMS_BUCKET", "xyme-data-dev"
)


def download_single_s3_file(object_key: str, dest: Path):
    s3 = boto3.client("s3")

    if dest.exists():
        logger.info(f"Skipping existing file: {dest}")
        return

    logger.info(f"Downloading {object_key} from S3...")
    s3.download_file(
        LIGANDMPNN_MODEL_PARAMS_BUCKET,
        f"protein-science/LigandMPNN_models/{object_key}",
        str(dest),
    )
    logger.debug(f"Downloaded: {dest}")


def download_model_parameters(model_parameter_dir: Path = MODEL_PARAMS_DIR) -> None:
    """
    Download LigandMPNN and ProteinMPNN models from AWS S3.
    """

    def destination_path(object_key: str) -> Path:
        return model_parameter_dir / object_key

    object_keys = [Path(uri).name for uri in parameter_file_uris]

    Parallel(n_jobs=multiprocessing.cpu_count() * 2, prefer="threads")(
        delayed(download_single_s3_file)(key, destination_path(key))
        for key in object_keys
    )
