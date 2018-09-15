import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

#rect = Rectangle((-2,-2),4,2, facecolor="none", edgecolor="none", fill=False)
rect = Rectangle((-2,-2),4,2, fill=False, ec="b")
circle = Circle((0,0),1, fill=False)

ax = plt.axes()

ax.add_patch(rect)
ax.add_patch(circle)

circle.set_clip_path(rect)

plt.ion()
plt.axis('equal')
plt.axis((-3,3,-3,3))
plt.show()
