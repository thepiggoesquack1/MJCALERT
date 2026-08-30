"""Package the maintained wheel distribution that provides ``webrtcvad``."""

from PyInstaller.utils.hooks import copy_metadata

datas = copy_metadata("webrtcvad-wheels")
