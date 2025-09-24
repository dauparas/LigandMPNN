import asyncio
from pathlib import Path

import aiofiles
import aiohttp
from aiohttp import ClientSession
from botocore.exceptions import ClientError, NoCredentialsError
from loguru import logger
from tqdm.asyncio import tqdm

from ligandmpnn.params.common import Source, parameter_file_uris
from ligandmpnn.paths import MODEL_PARAMS_DIR


async def download_file(
    session: ClientSession, url: str, output_path: Path, chunk_size: int = 8192
) -> Path:
    """
    Download a file from URL using aiohttp with streaming to minimize memory usage.
    """
    if output_path.exists():
        logger.debug(f"File {output_path} already exists, skipping download")
        return output_path

    async with session.get(url) as response:
        # Check if request was successful
        response.raise_for_status()

        # Get content length if available
        content_length = response.headers.get("Content-Length")
        if content_length:
            total_size = int(content_length)
            logger.debug(f"File size: {total_size / (1024 * 1024):.2f} MB")
        else:
            total_size = None
            logger.debug("File size: Unknown")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Stream the file to disk
        downloaded = 0

        async with aiofiles.open(output_path, "wb") as file:
            async for chunk in response.content.iter_chunked(chunk_size):
                await file.write(chunk)
                downloaded += len(chunk)

                # Print progress if we know the total size
                if total_size:
                    progress = (downloaded / total_size) * 100
                    logger.debug(
                        f"\rProgress: {progress:.1f}% ({downloaded / (1024 * 1024):.2f} MB)"
                    )
                else:
                    logger.debug(f"\rDownloaded: {downloaded / (1024 * 1024):.2f} MB")

        return output_path


async def fetch_params(model_parameters_dir: Path = MODEL_PARAMS_DIR) -> list[Path]:
    """
    This method will fetch the parameters from the canonical
    fileserver at https://files.ipd.uw.edu
    """
    session = aiohttp.ClientSession()

    def get_file_path(uri: str) -> Path:
        return Path(model_parameters_dir, uri.rsplit("/", maxsplit=1)[-1])

    fns = [
        download_file(session, uri, output_path=get_file_path(uri), chunk_size=8192)
        for uri in parameter_file_uris
    ]

    logger.info("Checking parameter files")
    results = await tqdm.gather(*fns)
    await session.close()
    logger.info(f"Parameters set in {model_parameters_dir}")

    return results


def fetch_params_sync(model_parameters_dir: Path = MODEL_PARAMS_DIR) -> list[Path]:
    """
    A convenience sync version of fetch_original_uw_params
    """
    return asyncio.run(fetch_params(model_parameters_dir))


if __name__ == "__main__":
    asyncio.run(fetch_params())
