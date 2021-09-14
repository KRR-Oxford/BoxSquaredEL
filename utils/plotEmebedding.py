import cv2

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

rect1 = plt.Rectangle((0.4360, 0.4864),0.09,-0.01, fill=False,edgecolor='0.1')
#rect2 = plt.Rectangle((0.2,0.2),0.65,0.4, fill=False,edgecolor='0.1')
ax.add_patch(rect1)
#ax.add_patch(rect2)

plt.show()