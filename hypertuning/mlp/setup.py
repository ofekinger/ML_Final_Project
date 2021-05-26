from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['cloudml-hypertune']

setup(
    name='rf_hp_tuning',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='HP tuning for binary classification using RF'
    )
