from pathlib import Path

from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.errors import FileError


class WheelWithoutDotenv(bdist_wheel):
    def write_wheelfile(self, wheelfile_base, *args, **kwargs):
        # Editable builds reuse this hook for metadata outside the wheel payload.
        if Path(wheelfile_base).parent.resolve() == Path(self.bdist_dir).resolve():
            for path in Path(self.bdist_dir).rglob("*"):
                if path.is_file() and (path.name.startswith(".env") or path.name.endswith(".env")):
                    raise FileError(
                        "Refusing wheel containing dotenv config; "
                        "clean stale build output and rebuild."
                    )
        return super().write_wheelfile(wheelfile_base, *args, **kwargs)
