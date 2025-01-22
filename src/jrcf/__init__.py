from pathlib import Path

import jpype
import jpype.imports
from jpype.types import *  # type: ignore [reportWildcardImportFromLibrary] # noqa: F403

from .__version__ import __version__

here = Path(__file__).parent
libs = here / "lib" / "*"

jpype.addClassPath(str(libs))
jpype.startJVM(convertStrings=False)


def java_gc():
    jpype.java.lang.System.gc()


__all__ = ["__version__", "java_gc"]
