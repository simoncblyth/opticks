import numpy as np
import pyvista as pv

# NEW (Works across most PyVista versions by leveraging Matplotlib)
import matplotlib as mpl
cmap_object = mpl.colormaps.get_cmap("turbo")



# ------------------------------------------------------------
# Your data (copy-paste from your output)
# Format: (N, 2, 3) -> N boxes, each with [corner_min, corner_max]
# ------------------------------------------------------------
boxes = np.array([
    [[ -5990.862, -33975.867,  18935.   ], [ -5990.862, -33975.867,  18935.   ]],
    [[ -5990.862, -33975.867,  18935.   ], [ -5990.862, -33975.867,  18935.   ]],
    [[ -5990.862, -33975.867,  18935.   ], [ -5990.862, -33975.867,  18935.   ]],
    [[ -4215.429, -16817.785,  18700.   ], [ -4138.971, -16741.327,  19170.   ]],
    [[ -3869.297, -16878.817,  17851.472], [ -3792.839, -16802.359,  18321.472]],
    [[ -3033.66 , -17026.163,  17500.   ], [ -2957.202, -16949.705,  17970.   ]],
    [[ -2198.023, -17173.508,  17851.472], [ -2121.565, -17097.05 ,  18321.472]],
    [[ -1851.891, -17234.541,  18700.   ], [ -1775.433, -17158.083,  19170.   ]],
    [[ -2198.023, -17173.508,  19548.528], [ -2121.565, -17097.05 ,  20018.528]],
    [[ -3033.66 , -17026.163,  19900.   ], [ -2957.202, -16949.705,  20370.   ]],
    [[ -3869.297, -16878.817,  19548.528], [ -3792.839, -16802.359,  20018.528]]
])


boxes = boxes[:]   # all
#boxes = boxes[:3] # only degenerate
#boxes = boxes[3:] # exclude degenerate


N = len(boxes)



# If you already have the array in your session, you can just do:
# boxes = f.stree__get_frame_global_LVID_124.reshape(-1, 2, 3)

# ------------------------------------------------------------
# Create a PyVista plotter
# ------------------------------------------------------------
plotter = pv.Plotter(window_size=(1200, 900))
plotter.set_background("black")

# Colors for variety (you can change the colormap if you want)
#colors = pv.get_cmap("turbo", N)
colors = (cmap_object(np.linspace(0, 1, N)) * 255).astype(np.uint8)
# Use colors[i][:3] for RGB tuple in your loop

#
#
#      (x1,y2,z2)      (x2,y2,z2)
#           7-----------6
#           |           |
#     4------------5    |
#     |     |      |    |
#     |     |      |    |
#     |     3------|----2 (x2,y2,z1)
#     |            |
#     0------------1
#  (x1,y1,z1)    (x2,y1,z1)
#
#     Z
#     |  Y
#     | /
#     |/
#     +-----> X
#

for i, box in enumerate(boxes):
    # Extract the two opposite corners
    c1 = box[0]
    c2 = box[1]

    # Build the 8 corners of the axis-aligned bounding box
    x1, y1, z1 = c1
    x2, y2, z2 = c2
    corners = np.array([
        [x1, y1, z1],     # 0
        [x2, y1, z1],     # 1  +X
        [x2, y2, z1],     # 2  +Y
        [x1, y2, z1],     # 3  -X
        [x1, y1, z2],     # 4
        [x2, y1, z2],     # 5  +X
        [x2, y2, z2],     # 6  +Y
        [x1, y2, z2],     # 7  -X
    ])

    # Create a PolyData with the 8 points
    cloud = pv.PolyData(corners)

    # Define the 12 edges (lines) of the box
    edges = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ])

    N_lines = edges.shape[0]
    count_col = np.full((N_lines, 1), 2)  # Creates a column of 12 twos
    lines = np.hstack([count_col, edges]).ravel()


    cloud.lines = lines

    #color = colors[i][:3]
    color = "red"

    # Add to plotter
    plotter.add_mesh(
        cloud,
        color=color,
        line_width=4,
        name=f"bbox_{i}",
        label=f"BBox {i}"
    )

    # Optional: add a small sphere at the "min" corner for orientation
    plotter.add_mesh(
        pv.Sphere(radius=150, center=c1),
        color="yellow",
        opacity=0.8
    )

# Add axes, legend, and make it pretty
plotter.add_axes(
    xlabel='X (mm?)', ylabel='Y (mm?)', zlabel='Z (mm?)',
    line_width=6, labels_off=False
)
plotter.show_grid()
plotter.add_legend()

# Camera position that works well for your coordinate range
plotter.camera_position = 'iso'
plotter.camera.zoom(1.2)

print("Close the window to exit.")
#plotter.show(cpos="xy")  # or just plotter.show() for interactive view
plotter.show()  # or just plotter.show() for interactive view



