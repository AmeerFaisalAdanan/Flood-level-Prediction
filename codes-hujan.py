import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
x_rainfall = np.arange(0, 300, 1)
rainfall_low = fuzz.trimf(x_rainfall, [0, 61, 1])
rainfall_low = fuzz.trimf(x_rainfall, [0, 0, 61])
rainfall_moderate = fuzz.trimf(x_rainfall, [61, 183, 183])
rainfall_high = fuzz.trimf(x_rainfall, [183, 200, 300])
fig, ax = plt.subplots()
ax.set_title('Jumlah Hujan')
ax.plot(x_rainfall, rainfall_low, linewidth=1.5, label='low')
ax.plot(x_rainfall, rainfall_moderate, linewidth=1.5, label='moderate')
ax.plot(x_rainfall, rainfall_high, linewidth=1.5, label='heavy')
ax.legend()
plt.show()