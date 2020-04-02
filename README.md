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
See CuPy [reference section](https://github.com/cupy/cupy#reference).
