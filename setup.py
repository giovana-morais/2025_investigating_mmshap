from setuptools import setup, find_packages

setup(
    name="investigate_mmshap",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Force rename Qwen-Audio to Qwen_Audio in the
    # package namespace
    py_modules=[pkg.replace("-", "_") for pkg in
        find_packages(where="src")],
    )
