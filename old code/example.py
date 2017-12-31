import numpy as np
import skfuzzy as fuzz
import matplotlib .pyplot as plt

CURRENT_TEMP = 18
CURRENT_HUM = 50

x_temp = np.arange(-20, 41, 1)
x_hum = np.arange(0, 101, 1)
x_sys = np.arange(12, 36, 1)

print("x_temp: " + str(x_temp))
print("x_hum: " + str(x_hum))
print("x_sys: " + str(x_sys))


# Generate fuzzy membership functions
temp_lo = fuzz.trimf(x_temp, [-20, -20, 5])
print("temp_lo: " + str(temp_lo))
temp_md = fuzz.trimf(x_temp, [0, 10, 20])
print("temp_md: " + str(temp_md))
temp_hi = fuzz.trimf(x_temp, [15, 40, 40])
print("temp_hi: " + str(temp_hi))


fig, ax = plt.subplots()
ax.set_title('Temperature')
ax.plot(x_temp, temp_lo, linewidth=1.5, label='low')
ax.plot(x_temp, temp_md, linewidth=1.5, label='medium')
ax.plot(x_temp, temp_hi, linewidth=1.5, label='high')
ax.legend()
plt.axis([-20, 40, 0, 1])
plt.show()



hum_lo = fuzz.trimf(x_hum, [0, 0, 50])
print("hum_lo: " + str(hum_lo))
hum_md = fuzz.trimf(x_hum, [30, 45, 60])
print("hum_md: " + str(hum_md))
hum_hi = fuzz.trimf(x_hum, [55, 100, 100])
print("hum_hi: " + str(hum_hi))


sys_cool2 = fuzz.trimf(x_sys, [12, 12, 15])
print("sys_cool2: " + str(sys_cool2))
sys_cool1 = fuzz.trimf(x_sys, [14, 15.5, 17])
print("sys_cool1: " + str(sys_cool1))
sys_off = fuzz.trimf(x_sys, [16, 18.5, 21])
print("sys_off: " + str(sys_off))
sys_heat1 = fuzz.trimf(x_sys, [20, 24, 28])
print("sys_heat1: " + str(sys_heat1))
sys_heat2 = fuzz.trimf(x_sys, [26, 35, 35])
print("sys_heat2: " + str(sys_heat2))


# We need the activation of our fuzzy membership functions at these values.
# The exact values temp=36 and hum=7.2 do not exist on our universes...
# This is what fuzz.interp_membership exists for!
temp_level_lo = fuzz.interp_membership(x_temp, temp_lo, CURRENT_TEMP)
print("temp_level_lo: " + str(temp_level_lo))
temp_level_md = fuzz.interp_membership(x_temp, temp_md, CURRENT_TEMP)
print("temp_level_md: " + str(temp_level_md))
temp_level_hi = fuzz.interp_membership(x_temp, temp_hi, CURRENT_TEMP)
print("temp_level_hi: " + str(temp_level_hi))

hum_level_lo = fuzz.interp_membership(x_hum, hum_lo, CURRENT_HUM)
print("hum_level_lo: " + str(hum_level_lo))
hum_level_md = fuzz.interp_membership(x_hum, hum_md, CURRENT_HUM)
print("hum_level_md: " + str(hum_level_md))
hum_level_hi = fuzz.interp_membership(x_hum, hum_hi, CURRENT_HUM)
print("hum_level_hi: " + str(hum_level_hi))

results = dict()

# Now we take our rules and apply them.
# Rule 1: temp_lo -> heat2
sys_activation_heat2 = np.fmin(temp_level_lo, sys_heat2)
active_rule1 = temp_level_lo
print("\nactive_rule1: " + str(active_rule1))
print("sys_activation_heat2: " + str(sys_activation_heat2))
results['active_rule1'] = active_rule1

# Rule 2: temp_md & hum_hi -> heat1
active_rule2 = np.fmin(temp_level_md, hum_level_hi)
print("\nactive_rule2: " + str(active_rule2))

sys_activation_heat1 = np.fmin(active_rule2, sys_heat1)
print("sys_activation_heat1: " + str(sys_activation_heat1))
results['active_rule2'] = active_rule2

# Rule 3: temp_md & (hum_md || hum_lo) -> off
hum_combined = np.fmax(hum_level_md, hum_level_lo)  # OR
print("\nhum_combined: " + str(hum_combined))
active_rule3 = np.fmin(temp_level_md, hum_combined)  # AND
#print("active_rule3: " + str(active_rule3))
sys_activation_off = np.fmin(active_rule3, sys_off)
print("sys_activation_off: " + str(sys_activation_off))
results['active_rule3'] = active_rule3

# Rule 4: temp_hi & hum_lo -> cool1

active_rule4 = np.fmin(temp_level_hi, hum_level_lo)  # AND
#print("\nactive_rule4: " + str(active_rule4))

sys_activation_cool1 = np.fmin(active_rule4, sys_cool1)
print("sys_activation_cool1: " + str(sys_activation_cool1))
results['active_rule4'] = active_rule4

# Rule 5: temp_hi & (hum_md || hum_hi) -> cool2
hum_combined = np.fmax(hum_level_md, hum_level_hi)  # OR
print("\nhum_combined: " + str(hum_combined))
active_rule5 = np.fmin(temp_level_hi, hum_combined)  # AND
#print("active_rule5: " + str(active_rule5))
sys_activation_cool2 = np.fmin(active_rule5, sys_cool2)
print("sys_activation_cool2: " + str(sys_activation_cool2))
results['active_rule5'] = active_rule5

sys0 = np.zeros_like(x_sys)
print("sys0: " + str(sys0))

# Aggregate all five output membership functions together
aggregated = np.fmax(sys_activation_cool2, np.fmax(sys_activation_cool1, np.fmax(sys_activation_heat2,
                     np.fmax(sys_activation_heat1, sys_activation_off))))
print("\naggregated: " + str(aggregated))


# Calculate defuzzified result
sys_temp = fuzz.defuzz(x_sys, aggregated, 'centroid')
print("\nsys_temp: " + str(sys_temp))

sys_temp2 = fuzz.defuzz(x_sys, aggregated, 'bisector')
print("\nsys_temp2: " + str(sys_temp2))

sys_temp3 = fuzz.defuzz(x_sys, aggregated, 'lom')
print("\nsys_temp3: " + str(sys_temp3))

sys_temp4 = fuzz.defuzz(x_sys, aggregated, 'som')
print("\nsys_temp4: " + str(sys_temp4))

print(results)