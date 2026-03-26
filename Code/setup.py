from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "dfa_cpp",
        ["dfa.cpp"],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="dfa_cpp",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
