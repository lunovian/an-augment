from setuptools import setup, find_packages

setup(
    name='MedAugment',
    version='0.1',
    description='A Medical Imaging Augmentation Library',
    author='lunovian',
    license='Apache-2.0',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=['numpy', 'scipy', 'opencv-python'],
)
