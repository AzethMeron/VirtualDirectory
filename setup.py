from setuptools import setup, find_packages

setup(
    name='VirtualDirectory',
    version='1.3.0',
    packages=find_packages(),
    install_requires=[
        'tqdm>=4.66.5',
        'opencv-python>=4.10.0.84',
        'pillow>=10.4.0',
        'numpy>=2.1.0',
    ],
)
