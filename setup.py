from setuptools import setup, find_packages

setup(
    name='Sapeon-Frontend-Compiler',
    version='0.1',
    package_dir={'': 'src'},
    install_requires=[
        'onnx >= 0.12.0',
    ]
)