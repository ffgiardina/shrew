from setuptools import setup

setup(
    name="pyshrew",
    version="0.1",
    description="A Python extension module for the shrew library",
    author="ffg",
    author_email="",
    url="https://github.com/ffgiardina/shrew",
    packages=["pyshrew"],
    package_data={
        "pyshrew": ["*.so"], 
    },
    include_package_data=True,
    zip_safe=False,
)
