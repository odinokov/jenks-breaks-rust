import numpy as np
import requests
import json
import jenks_breaks

url = "https://raw.githubusercontent.com/mthh/jenkspy/master/tests/test.json"

response = requests.get(url)

response.raise_for_status()

data = json.loads(response.text)

num_classes = 5

data_array = np.sort(data)

breaks = jenks_breaks.jenks_breaks_optimized(data_array, num_classes)

# Output the result
print("Breaks indices:")
print(*breaks, sep=", ")
print("Breaks values:")
print(data_array[0], *data_array[np.array(breaks)], data_array[-1], sep=', ')
print("Expected:")
print("0.0028109620325267315, 2.0935479691252112, 4.205495140049607, 6.178148351609707, 8.09175917180255, 9.997982932254672]")
print("^                      ^                    ^                 ^                  ^                 ^")
print("Lower bound            Upper bound          Upper bound       Upper bound        Upper bound       Upper bound")
print("1st class              1st class            2nd class         3rd class          4th class         5th class")
print("(Minimum value)                                                                                    (Maximum value)")