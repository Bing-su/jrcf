from pathlib import Path

import jpype
import jpype.imports
from jpype.types import *  # type: ignore [reportWildcardImportFromLibrary] # noqa: F403

here = Path(__file__).parent
libs = here / "lib" / "*"

jpype.addClassPath(str(libs))
jpype.startJVM(convertStrings=False)
