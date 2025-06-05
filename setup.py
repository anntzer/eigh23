from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


setup(ext_modules=[
    Pybind11Extension("eigh23._eigh23", ["ext/_eigh23.cpp"], cxx_std=17)])
