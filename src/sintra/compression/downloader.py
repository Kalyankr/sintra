"""Model downloading utilities using HuggingFace Hub.

Handles downloading, caching, and locating HuggingFace models.
"""

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sintra"


class DownloadError(Exception):
    """Raised when model download fails."""
    pass


class ModelDownloader:
    """Downloads and caches HuggingFace models.
    
    Models are cached in ~/.cache/sintra/downloads/ to avoid re-downloading.
    
    Example:
        >>> downloader = ModelDownloader()
        >>> path = downloader.download("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        >>> print(path)
        /home/user/.cache/sintra/downloads/TinyLlama--TinyLlama-1.1B-Chat-v1.0
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the downloader.
        
        Args:
            cache_dir: Base cache directory. Defaults to ~/.cache/sintra/
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.downloads_dir = self.cache_dir / "downloads"
        self.downloads_dir.mkdir(parents=True, exist_ok=True)
        self._api = HfApi()
    
    def download(
        self,
        model_id: str,
        revision: str = "main",
        token: Optional[str] = None,
    ) -> Path:
        """Download a model from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID (e.g., "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            revision: Git revision (branch, tag, or commit hash)
            token: HuggingFace token for gated models
            
        Returns:
            Path to the downloaded model directory
            
        Raises:
            DownloadError: If download fails (not found, gated, network error)
        """
        logger.info(f"Downloading model: {model_id}")
        
        # Convert model_id to safe directory name
        safe_name = model_id.replace("/", "--")
        local_dir = self.downloads_dir / safe_name
        
        # Check if already downloaded
        if self._is_complete_download(local_dir):
            logger.info(f"Model already cached at {local_dir}")
            return local_dir
        
        try:
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                local_dir=local_dir,
                token=token,
                # Only download model weights, skip large unnecessary files
                ignore_patterns=[
                    "*.md",
                    "*.txt", 
                    ".gitattributes",
                    "original/",  # Skip original pytorch bins if safetensors exist
                ],
            )
            logger.info(f"Downloaded to {local_dir}")
            return local_dir
            
        except RepositoryNotFoundError:
            raise DownloadError(
                f"Model '{model_id}' not found on HuggingFace Hub. "
                "Check the model ID at https://huggingface.co/models"
            )
        except GatedRepoError:
            raise DownloadError(
                f"Model '{model_id}' is gated. "
                "Accept the license at https://huggingface.co/{model_id} "
                "and provide a token with --hf-token"
            )
        except Exception as e:
            raise DownloadError(f"Failed to download '{model_id}': {e}") from e
    
    def _is_complete_download(self, path: Path) -> bool:
        """Check if a model is fully downloaded.
        
        Verifies config.json and at least one weight file exists.
        """
        if not path.exists():
            return False
        
        has_config = (path / "config.json").exists()
        has_weights = (
            list(path.glob("*.safetensors")) or
            list(path.glob("*.bin")) or
            list(path.glob("*.pt"))
        )
        
        return has_config and bool(has_weights)
    
    def get_model_info(self, model_id: str) -> dict:
        """Get model metadata from HuggingFace Hub.
        
        Returns:
            Dict with model info (downloads, likes, tags, etc.)
        """
        try:
            info = self._api.model_info(model_id)
            return {
                "id": info.id,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
            }
        except Exception as e:
            logger.warning(f"Could not fetch model info: {e}")
            return {}
    
    def list_cached_models(self) -> list[Path]:
        """List all models in the cache."""
        if not self.downloads_dir.exists():
            return []
        return [p for p in self.downloads_dir.iterdir() if p.is_dir()]


def download_model(
    model_id: str,
    cache_dir: Optional[Path] = None,
    token: Optional[str] = None,
) -> Path:
    """Convenience function to download a model.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Optional cache directory
        token: Optional HuggingFace token
        
    Returns:
        Path to downloaded model
    """
    downloader = ModelDownloader(cache_dir=cache_dir)
    return downloader.download(model_id, token=token)
