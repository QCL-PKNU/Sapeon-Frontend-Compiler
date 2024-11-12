import numpy as np
from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from os import path

package_name = "sapeon.simulator"

version_number = ""
with open("release_version") as f:
    version_number = f.readline().strip()

BASE_DIR = path.dirname(path.abspath(__file__))
lib_dir = path.join(BASE_DIR, "build")

ext_modules = [
    Pybind11Extension(
        package_name,
        sorted(glob("python/*.cpp")),
        include_dirs=["include", np.get_include()],
        libraries=["spsim"],
        library_dirs=["build"],
        extra_link_args=["-Wl,-rpath", lib_dir],
        language="c++",
    ),
]

install_requires = [
    "numpy",
]

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: Other/Proprietary License",
    "Operating System : POSIX :: Linux",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3.8",
]

setup(
    name=package_name,
    version=version_number,
    description="sapeon.simulator is an simulator for SAPEON device",
    url="https://www.sapeon.com",
    author="Sapeon Inc.",
    license="Proprietary",
    install_requires=install_requires,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=classifiers,
    python_requires=">=3.8",
)
