import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='colorito',
     version='0.1.0',
     scripts=[],
     author="Federico Giannoni",
     author_email="federico.giannoni.home@gmail.com",
     description="A Python library for color search",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/kekgle/colorito",
     python_requires=">=3.0, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
     packages=setuptools.find_packages(),
     # entry_points={"console_scripts": [
     #   'cmd = package.file:function'
     # ]},
     install_requires=[
         "tqdm",
         "sklearn",
         "numpy",
         "scipy",
         "torch",
         "spacy",
         "colormath",
         "matplotlib"
     ],
     classifiers=[
         "Programming Language :: Python :: 3.5",
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Operating System :: OS Independent",
         "Development Status :: 3 :: Alpha",
         "Topic :: Scientific/Engineering"
     ],
 )