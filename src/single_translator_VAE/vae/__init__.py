import logging
from importlib.metadata import version

from rich.console import Console
from rich.logging import RichHandler

from .vae_model import VAEModel, VAEModule

logger = logging.getLogger(__name__)
# set the logging level
logger.setLevel(logging.INFO)

# nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("Single Translator VAE: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# this prevents double outputs
logger.propagate = False

__all__ = ["vae_model", "vae_model"]

__version__ = version("single_translator_VAE")
