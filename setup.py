from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="structure-tensor",
      version="0.1.0",
      author="Vedrana Andersen Dahl, Niels Jeppesen",
      author_email="vand@dtu.dk, niejep@dtu.dk",
      description="Structure tensor for Python",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="",
      packages=["structure_tensor"],
      classifiers=[
          "Development Status :: 3 - Alpha",
          "Environment :: Console",
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: MIT License",
          "Natural Language :: English",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Topic :: Scientific/Engineering :: Image Recognition",
          "Topic :: Scientific/Engineering :: Artificial Intelligence",
          "Topic :: Scientific/Engineering :: Mathematics",
      ])
