from setuptools import setup, find_packages

setup(
    name="projet",  # Nom du projet
    version="0.1.0",  # Version du projet
    packages=find_packages(where='src'),  
    package_dir={'': 'src'},
)
