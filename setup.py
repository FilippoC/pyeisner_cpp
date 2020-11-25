from setuptools import setup, Extension

setup(
    name="pyeisner",
    version="0.1",
    include_dirs=["./cpp/"],
    ext_modules=[
        Extension(
            "pyeisner",
            sources=[
                "cpp/pyeisner.cpp",
                "cpp/marginals.cpp",
                "cpp/argmax.cpp",
            ],
            language='c++',
            #extra_link_args=["-fopenmp"],
            extra_compile_args=[
                '-std=c++11',
                '-Wfatal-errors',
                '-Wall',
                '-Wextra',
                '-pedantic',
                '-O3',
                '-funroll-loops',
                '-march=native',
                '-fPIC',
                #'-fopenmp'
            ],
        )
    ]
);

