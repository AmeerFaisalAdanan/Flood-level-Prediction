#importing support library and dependencies
import numpy as np
import skfuzzy as fuzz
import matplotlib .pyplot as plt


#setting up graph for membership value
x_riverlevel = np.arange(0, 6, 1)
riverlevel_normal = fuzz.trimf(x_riverlevel, [0,1,2])
riverlevel_berjaga = fuzz.trimf(x_riverlevel, [0,2,3])
riverlevel_amaran = fuzz.trimf(x_riverlevel, [1,3,4])
riverlevel_bahaya = fuzz.trimf(x_riverlevel, [3,4,5])
fig, ax = plt.subplots()
plt.axis([0, 5, 0, 1])#the graph lie perfectly on the axis
ax.set_title('River level')#graph title

#plotting membership value
ax.plot(x_riverlevel, riverlevel_normal,linewidth=1.5, label='normal')
ax.plot(x_riverlevel, riverlevel_berjaga,linewidth=1.5, label='berjaga')
ax.plot(x_riverlevel, riverlevel_amaran,linewidth=1.5, label='amaran')
ax.plot(x_riverlevel, riverlevel_bahaya,linewidth=1.5, label='bahaya')
ax.legend()
#display the graph windows
plt.show()



#Rules evaluation

