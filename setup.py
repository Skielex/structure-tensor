from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name="structure-tensor",
      version="0.2.0",
      author="Niels Jeppesen",
      author_email="niejep@dtu.dk",
      description="Fast and simple to use 2D and 3D structure tensor implementation for Python.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/Skielex/structure-tensor",
      packages=[
          "structure_tensor",
          "structure_tensor.cp",
      ],
      python_requires='>=3',
      install_requires=["numpy>=1.16", "scipy>=1.3"],
      extras_require={"CuPy": ["cupy>=8"]},
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
