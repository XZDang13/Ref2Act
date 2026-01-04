from setuptools import setup, find_packages

setup(
    name='Ref2Act',
    version='0.2.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "Ref2Act": ["assets/**/*"],
    },
    install_requires=[
        # Add your dependencies here
    ],
)
