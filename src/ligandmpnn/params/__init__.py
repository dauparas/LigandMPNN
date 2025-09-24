from pathlib import Path

from botocore.exceptions import NoCredentialsError, ClientError

from ligandmpnn.params.common import Source
from .washington_university import fetch_params_sync
from .xyme import download_model_parameters
from ligandmpnn.logger import logger
from ..paths import MODEL_PARAMS_DIR


def download_models(
    model_parameters_dir: Path = MODEL_PARAMS_DIR, source: Source = Source.xyme
) -> None:
    """
    Download LigandMPNN and ProteinMPNN models from specified sources, based on the order,
    the first will be tried first, and if it fails, the next one will be tried.
    The sources can be 'Xyme' which fetches the parameters from S3 or 'UW' for the University of Washington server

    :param model_parameters_dir: Directory to save the models. If None, defaults to LMPNN_MODEL_PATH.
    :param source: List of sources to download from. Options are 's3' and 'UW'.
    """

    if source == Source.xyme:
        try:
            xyme.download_model_parameters(model_parameters_dir)
        except NoCredentialsError:
            logger.error(
                "AWS credentials not found. Please configure your AWS credentials."
            )
        except ClientError as e:  # expired tokens
            logger.error(f"Failed to download models from S3: {e}")
        except KeyboardInterrupt:
            logger.info("Download interrupted by impatient user...")
    elif source == Source.washingtonUniversity:
        fetch_params_sync(model_parameters_dir)
    else:
        raise ValueError(
            f"Unknown source for LigandMPNN weights, expected one of {Source}"
        )
