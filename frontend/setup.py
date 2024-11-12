from setuptools import setup, find_packages


setup(
    name='Sapeon-Frontend-Compiler',
    version='1.0',
    description='Deep learning compiler',
    author='PKNU-QCL',
    package_dir={
        '', 'src'
    },
    packages=find_packages(),
    install_requires=[
        'numpy==1.26.0',
        'tensorflow==2.9.1',
    ],
    entry_points={
        'console_scripts': {
            'fcompiler=frontend.compiler::'
        }
    },
    python_required='>=3.9'
)