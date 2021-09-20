import cv2

import matplotlib.pyplot as plt
def plot_rec(recList):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for rec in recList:
        ax.add_patch(rec)





def calculate(center, offset):
    width = offset[0]-center[0]
    height = offset[1]-center[1]
    return width, height
