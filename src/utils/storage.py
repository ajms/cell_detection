from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@lru_cache
def get_project_root() -> Path:
    return Path.cwd()


@dataclass
class Storage:
    endpoint: str = ""

    def uri2path():
        pass


if __name__ == "__main__":
    print(get_project_root())
