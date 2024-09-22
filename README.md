# Rust version of Jenks Breaks for Python

**Fast implementation of the Jenks natural breaks (Fisher-Jenks) algorithm with Python bindings.**  
Compute "natural breaks" on sorted `numpy.ndarray` with dynamic programming.

The algorithm implemented by this library is also sometimes referred to as:
- **Fisher-Jenks algorithm**
- **Jenks Optimization Method**
- **Fisher exact optimization method**

This method calculates optimal class boundaries for numerical data, typically used in cartography and data classification.

## Installation

```bash
   git clone https://github.com/odinokov/jenks-breaks-rust.git
   pip install ./jenks-breaks-rust/dist/jenks_breaks-0.1.0-cp310-cp310-manylinux_2_34_x86_64.whl
   ```

## Building from Source

1. **Clone the repository:**

   ```bash
   git clone https://github.com/odinokov/jenks-breaks-rust.git
   cd jenks-breaks-rust
   ```

2. **Install `maturin`:**

   ```bash
   pip install maturin
   ```

3. **Build the library:**

   ```bash
   maturin develop --release
   ```

4. **Run the tests:**

   ```bash
   python test.py
   ```

## Usage

```python
import numpy as np
import requests
import json
from jenks_breaks import jenks_breaks_optimized

url = "https://raw.githubusercontent.com/mthh/jenkspy/master/tests/test.json"

response = requests.get(url)

response.raise_for_status()

data = json.loads(response.text)

num_classes = 5

data_array = np.sort(data)

breaks = jenks_breaks_optimized(data_array, num_classes)

# Output the result
print("Breaks indices:")
print(*breaks, sep=", ")
print("Breaks values:")
print(data_array[0], *data_array[np.array(breaks) - 1], data_array[-1], sep=', ')
print("Expected:")
print("0.0028109620325267315, 2.0935479691252112, 4.205495140049607, 6.178148351609707, 8.09175917180255, 9.997982932254672]")
print("^                      ^                   ^                  ^                  ^                 ^")
print("Lower bound            Upper bound          Upper bound       Upper bound        Upper bound       Upper bound")
print("1st class              1st class            2nd class         3rd class          4th class         5th class")
print("(Minimum value)                                                                                    (Maximum value)")
```

# Benchmarking Results

```python
from jenks_breaks import jenks_breaks_optimized
from jenkspy import JenksNaturalBreaks
import numpy as np

# Seed for reproducibility
np.random.seed(42)
data = np.sort(np.random.uniform(-1000, 1000, 10000))

# Benchmarking jenkspy: Fast Fisher-Jenks breaks for Python
# https://github.com/mthh/jenkspy

jnb = JenksNaturalBreaks(5)

%timeit jnb.fit(data)
# Output: 215 ms ± 16.3 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Benchmarking jenks_breaks (Rust)
%timeit jenks_breaks_optimized(data, 5)
# Output: 317 ms ± 6.72 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# Seed for reproducibility
np.random.seed(42)
data = np.sort(np.random.uniform(-10, 10, 100))

# Benchmarking jenkspy: Fast Fisher-Jenks breaks for Python
# https://github.com/mthh/jenkspy

%timeit jnb.fit(data)
# Output: 92.9 μs ± 2.89 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

# Benchmarking jenks_breaks (Rust)
%timeit jenks_breaks_optimized(data, 5)
# Output: 38.8 μs ± 1.62 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
```

## License

This library is licensed under the [MIT License](LICENSE).
