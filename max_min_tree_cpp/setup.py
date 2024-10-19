from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension


# 设置include_dirs
# pybind11的include
# `python3 -m pybind11 --includes` 进行查看
include_dirs = [
    './',
    '/home/ylgu/miniconda3/envs/dg/include/python3.9',
    '/home/ylgu/miniconda3/envs/dg/lib/python3.9/site-packages/pybind11/include'
    ]


ext_module = [
    Pybind11Extension(
    name='tree_trans3D',
    sources=['TreeTrans3D.cpp'],
    include_dirs=include_dirs,
    cxx_std=23,
    language='c++',
    # extra_compile_args=['-g', '-O0']
    extra_compile_args=["-O3","-fPIC"],
    )
]

setup(
    name='tree_trans3D',
    ext_modules=ext_module
)
