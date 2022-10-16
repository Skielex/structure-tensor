# Structure Tensor for Python
Fast and simple to use 2D and 3D [structure tensor](https://en.wikipedia.org/wiki/Structure_tensor) implementation for Python.

## Installation
Install package using ```pip install structure-tensor``` or clone the repository.

### CUDA Support
For CUDA support install extra (optional) dependancy [CuPy](https://github.com/cupy/cupy). If CUDA is installed on your system, ```pip install cupy``` should be enough, but may be slow as CuPy will compile code during install. Alternatively use one of the [precompiled packages](https://github.com/cupy/cupy#installation).

## Tiny Examples
The parameters for the structure tensor calculations are $\rho$ (```rho```) and $\sigma$ (```sigma```), which are scalar values.

### 2D and 3D using NumPy
The ```structure_tensor``` package support doing either 2D or 3D structure tensor analysis. Eigenvalues (```val```) are sorted acending.

``` python
import numpy as np
from structure_tensor import eig_special_2d, structure_tensor_2d

sigma = 1.5
rho = 5.5

# Load 2D data.
image = np.random.random((128, 128))

S = structure_tensor_2d(image, sigma, rho)
val, vec = eig_special_2d(S)
```

For volume with shape ```(x, y, z)``` the eigenvectors (```vec```) are returned as ```zyx```.

``` python
import numpy as np
from structure_tensor import eig_special_3d, structure_tensor_3d

sigma = 1.5
rho = 5.5

# Load 3D data.
volume = np.random.random((128, 128, 128))

S = structure_tensor_3d(volume, sigma, rho)
val, vec = eig_special_3d(S)
```

### 3D using CuPy
CuPy functions are available in the ```structure_tensor.cp``` module. They work similar to their NumPy counterparts, except that they return ```cupy.ndarray```s. The functions will automatically handle moving input data if necessary.

``` python
import cupy as cp
import numpy as np
from structure_tensor.cp import eig_special_3d, structure_tensor_3d

sigma = 1.5
rho = 5.5

# Load 3D data.
volume = np.random.random((128, 128, 128))

S = structure_tensor_3d(volume, sigma, rho)
val, vec = eig_special_3d(S)

# Convert from cupy to numpy. Moves data from GPU to CPU.
val = cp.asnumpy(val)
vec = cp.asnumpy(vec)
```

## Contributions
Contributions are welcome, just create an [issue](https://github.com/Skielex/structure-tensor/issues) or a [PR](https://github.com/Skielex/structure-tensor/pulls).

## More information
- [Wikipedia - Structure tensor](https://en.wikipedia.org/wiki/Structure_tensor)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [CuPy](https://cupy.chainer.org/)

## License
MIT License (see LICENSE file).

## Reference
If you use this any of this for academic work, please consider citing our work.

### Initial paper
> Jeppesen, N., et al. "Characterization of the fiber orientations in non-crimp glass fiber reinforced composites using structure tensor." *IOP Conference Series: Materials Science and Engineering.* Vol. 942. No. 1. IOP Publishing, 2020.<br>
[ [paper](https://doi.org/10.1088/1757-899x/942/1/012037) ]
[ [data and notebooks](https://doi.org/10.5281/zenodo.3877521) ]

``` bibtex
@article{Jeppesen_2020,
	doi = {10.1088/1757-899x/942/1/012037},
	url = {https://doi.org/10.1088/1757-899x/942/1/012037},
	year = 2020,
	month = {oct},
	publisher = {{IOP} Publishing},
	volume = {942},
	number = {1},
	pages = {012037},
	author = {N Jeppesen and V A Dahl and A N Christensen and A B Dahl and L P Mikkelsen},
	title = {Characterization of the fiber orientations in non-crimp glass fiber reinforced composites using structure tensor},
	journal = {{IOP} Conference Series: Materials Science and Engineering},
}
```
### Other papers
> Jeppesen, N., et al. "Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis." *Composites Part A: Applied Science and Manufacturing* 149 (2021): 106541.<br>
[ [paper](https://doi.org/10.1016/j.compositesa.2021.106541) ]
[ [data and notebooks](https://doi.org/10.5281/zenodo.4446498) ]

>Auenhammer, Robert M., et al. "Robust numerical analysis of fibrous composites from X-ray computed tomography image data enabling low resolutions." *Composites Science and Technology* (2022): 109458.<br>
[ [paper](https://doi.org/10.1016/j.compscitech.2022.109458) ]
[ [data](https://doi.org/10.5281/zenodo.5774920) ]

>Auenhammer, Robert M., et al. "X-ray computed tomography data structure tensor orientation mapping for finite element models—STXAE." *Software Impacts* 11 (2022): 100216.<br>
[ [paper](https://doi.org/10.1016/j.simpa.2021.100216) ]

### Data and notebooks
>Jeppesen, N, Dahl, VA, Christensen, AN, Dahl, AB, & Mikkelsen, LP. (2020). Characterization of the Fiber Orientations in Non-Crimp Glass Fiber Reinforced Composites using Structure Tensor [Data set]. Zenodo. https://doi.org/10.5281/zenodo.3877522

>Jeppesen, N, Mikkelsen, Lars P., Dahl, V.A., Nymark, A.N., & Dahl, A.B. (2021). Quantifying effects of manufacturing methods on fiber orientation in unidirectional composites using structure tensor analysis [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4446499

>Auenhammer, R.M., Jeppesen, N, Mikkelsen, Lars P., Dahl, V.A., Blinzler, B.J., & Asp, L.E. (2021). X-ray computed tomography aided engineering approach for non-crimp fabric reinforced composites [Data set] [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5774920


### CuPy
See CuPy [reference section](https://github.com/cupy/cupy#reference).
