from setuptools import setup, find_packages

setup(
    name='D3Fold',
    version='0.1',
    packages=find_packages(),
    author='Dylan Abramson',
    author_email='dylanabramson33@gmail.com',
    description='A framework for easily implementing 3D protein models.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dylanabramson33/D3Fold',
    include_package_data=True,
)