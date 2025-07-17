import os
from pathlib import Path
from shlex import split

import jpype
import jpype.imports
from jpype.types import *  # type: ignore [reportWildcardImportFromLibrary] # noqa: F403

from .__version__ import __version__

here = Path(__file__).parent
libs = here / "lib" / "*"
java_opts = split(os.environ.get("JAVA_OPTS", ""))

jpype.addClassPath(str(libs))
jpype.startJVM(*java_opts, convertStrings=False)


def java_gc():
    """
    Trigger the Java garbage collector.

    This function calls the garbage collector of the Java Virtual Machine (JVM)
    using the JPype library. It invokes the `gc` method on the `System` class
    from the `java.lang` package to request garbage collection.
    """
    jpype.java.lang.System.gc()


__all__ = ["__version__", "java_gc"]
