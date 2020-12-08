import mph
import numpy as np
client = mph.Client()


filename = 'rus_srtio3_cube.mph'
model = client.load(filename)

## Modifying
model.parameter('nb_freq', '25')


model.solve('resonances')

freqs = model.evaluate('abs(freq)', 'MHz')
index_ok = freqs > 1e-4 # in MHz
freqs = freqs[index_ok]

for n, freq in enumerate(freqs):
    print(str(n+1) + ": " + str(np.round(freq, 3)) + " MHz")

## Close COMSOL file without saving solutions in the file
client.clear()