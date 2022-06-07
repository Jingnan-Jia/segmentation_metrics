import setuptools

def parse_requirements(requirements):
    with open(requirements) as f:
        return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

# reqs = parse_requirements('requirements.txt')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="seg_metrics", # Replace with your own username
    version="1.0.29",
    author="Jingnan",
    author_email="jiajingnan2222@gmail.com",
    description="A package to compute different segmentation metrics for Medical images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ordgod/segmentation_metrics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pandas', 'numpy', 'coverage', 'matplotlib', 'parameterized', 'tqdm', 'medutils', 'PySimpleGUI', 'SimpleITK'])
