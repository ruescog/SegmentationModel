from setuptools import find_packages, setup

setup(
    name='segmentationmodel',
    packages=find_packages(include=['segmentationmodel']),
    version='0.1.0',
    description='Segmentation Facade for Semtorch, Segmentron, Smp and mmsegmentation python libraries.',
    author='Rubén Escobedo Gutiérrez',
    license='MIT',
    install_requires=[
        "segmentation_models_pytorch",
        "semtorch",
        "GitPython",
        "PrettyTable",
        "mmcv-full",
        "torch==1.6.0+cu101",
        "torchvision==0.7.0+cu101"
        ],
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
        "https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html"
    ]
)