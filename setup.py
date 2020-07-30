from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='weak_labeling',
    version='0.1.0',
    description='weak labling experiments',
    long_description=readme,
    author='Collins Nji',
    mentor='Jennifer Melot',
    author_email='cn498@georgetown.edu',
    url='https://github.com/georgetown-cset/weak-labeling-experiments',
    license=license,
    packages=find_packages(exclude=('tests', 'documentation'))
)
