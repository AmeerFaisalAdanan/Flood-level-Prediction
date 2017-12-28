import numpy as np
import skfuzzy as fuzz
import matplotlib .pyplot as plt


CURRENT_RIVER_LEVEL = 1.82
CURRENT_RAINFALL = 0
x_riverlevel = np.arange(0, 6, 1)
x_rainfall = np.arange(0, 300, 1)
x_floodstatus = np.arange(0, 11, 1)
print("x_riverlevel: " + str(x_riverlevel))
print("x_rainfall: " + str(x_rainfall))
print("x_floodstatus"+ str(x_floodstatus))


#generating fuzzy membership value

#river membership value
riverlevel_normal = fuzz.trimf(x_riverlevel, [0,1,2])
print("riverlevel_normal: " + str(riverlevel_normal))
riverlevel_berjaga = fuzz.trimf(x_riverlevel, [0,2,3])
print("riverlevel_berjaga: " + str(riverlevel_berjaga))
riverlevel_amaran = fuzz.trimf(x_riverlevel, [1,3,4])
print("riverlevel_amaran: " + str(riverlevel_amaran))
riverlevel_bahaya = fuzz.trimf(x_riverlevel, [3,4,5])
print("riverlevel_bahaya: " + str(riverlevel_bahaya))


#rainfall membership value
rainfall_low = fuzz.trimf(x_rainfall, [0, 30, 61])
print("rainfall_low: " + str(rainfall_low))
rainfall_moderate = fuzz.trimf(x_rainfall, [40, 122, 183])
print("rainfall_moderate: " + str(rainfall_moderate))
rainfall_high = fuzz.trimf(x_rainfall, [160, 200, 300])
print("rainfall_high: " + str(rainfall_high))

#floodstatus membership value
flood_none = fuzz.trimf(x_floodstatus, [0, 1, 1])
print("flood_none: " + str(flood_none))
flood_minor = fuzz.trimf(x_floodstatus, [0.5, 3, 4])
print("flood_minor: " + str(flood_minor))
flood_moderate = fuzz.trimf(x_floodstatus, [3, 5, 6])
print("flood_moderate: " + str(flood_moderate))
flood_major = fuzz.trimf(x_floodstatus, [5.5, 7, 8])
print("flood_major: " + str(flood_major))
flood_extreme = fuzz.trimf(x_floodstatus, [7.5, 9, 10])
print("flood_extreme: " + str(flood_extreme))

#visualization of the membership value
#River level
fig, ax = plt.subplots()
ax.set_title('River Level')
ax.plot(x_riverlevel, riverlevel_normal,linewidth=1.5, label='normal')
ax.plot(x_riverlevel, riverlevel_berjaga,linewidth=1.5, label='berjaga')
ax.plot(x_riverlevel, riverlevel_amaran,linewidth=1.5, label='amaran')
ax.plot(x_riverlevel, riverlevel_bahaya,linewidth=1.5, label='bahaya')
ax.legend()
plt.axis([0, 5, 0, 1])
plt.show()

#Rainfall
fig, ax = plt.subplots()
ax.set_title('Rainfall')
ax.plot(x_rainfall, rainfall_low, linewidth=1.5, label='low')
ax.plot(x_rainfall, rainfall_moderate, linewidth=1.5, label='moderate')
ax.plot(x_rainfall, rainfall_high, linewidth=1.5, label='heavy')
ax.legend()
plt.axis([0, 300, 0, 1])
plt.show()

#Flood Status
fig, ax = plt.subplots()
ax.set_title('Flood Status')
ax.plot(x_floodstatus, flood_none, linewidth=1.5, label='No Flood')
ax.plot(x_floodstatus, flood_minor, linewidth=1.5, label='Minor Flood')
ax.plot(x_floodstatus, flood_moderate, linewidth=1.5, label='Moderate Flood')
ax.plot(x_floodstatus, flood_major, linewidth=1.5, label='Major Flood')
ax.plot(x_floodstatus, flood_extreme, linewidth=1.5, label='Extreme Flood')
ax.legend()
plt.axis([0, 11, 0, 1])
plt.show()