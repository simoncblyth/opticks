#!/usr/bin/env python
"""
tilted_rings_2.py
===================

Create ELU flight paths::

    ~/o/sysrap/tests/tilted_rings_2.py
    SLANTED=1 ~/o/sysrap/tests/tilted_rings_2.py

Use the ELU flight paths to control ifly animations::

    VIEW=/tmp/tilted_rings_2.npy             cxr_min.sh
    VIEW=/tmp/tilted_rings_2_horizontals.npy cxr_min.sh


Length units in the below. 
--------------------------

* scaling by 1000 converts distances in m to standard unit of mm
* division by 20000 is done assuming a MOI extent setting like::

   0,0,0,20000


"""

import os
import numpy as np

def get_tilt_rotation():
    r"""
          ax   +Z
           \   |
            \  |
             \ |
              \|
               +
              /
             /
            k : rotation axis


    1. form EMF coil tilt axis unit vector from theta phi angles
    2. pick rotation axis k from cross product of +Z "up" [0,0,1] and the tilt axis
    3. form tilt rotation matrix using Rodrigues' formula

       R = I + sin(angle)*K + (1-cos(angle))*K^2

    """
    # 1. form EMF coil tilt axis unit vector from theta phi angles

    theta = 56. * np.pi / 180.
    phi = -54. * np.pi / 180.
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    ax = np.array([st * cp, st * sp, ct])

    # 2. pick rotation axis k from cross product of +Z "up" [0,0,1] and the tilt axis
       
    k = np.array([ax[1], -ax[0], 0])
    k = k / np.linalg.norm(k)

    # 3. form tilt rotation matrix using Rodrigues' formula
    angle = np.arccos(ct)  # theta
    K = np.array([ [0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R



# SUP_Z_23
zs = np.fromstring("""
     21.452389, 21.081667, 20.531347, 19.594910, 18.240872, 16.518107,
     14.432318, 12.000134,  9.250595,  6.298513,  3.207950,  0.000000,
    -3.115456, -6.271698, -9.163020, -11.961896, -14.358739, -16.453920,
    -18.187246, -19.552710, -20.505130, -21.101293, -21.442995
 """, sep="," )*1000./20000.

# SUP_R_23
rs = np.fromstring("""
     2.026612,  4.005680,  6.322447,  8.822219, 11.377485, 13.822499,
    16.005342, 17.901379, 19.465059, 20.610428, 21.311241, 21.450000,
    21.324899, 20.618525, 19.506275, 17.500121, 16.071143, 13.918468,
    11.562004,  9.063568,  6.633064,  4.380809,  2.155714
 """, sep="," )*1000./20000.


zc = np.fromstring("""
        21.271,  20.602,  19.554,  18.555,
        17.077,  15.600,  14.122,  12.644,
        11.166,   9.685,   8.207,   6.729,
         5.251,   3.773,   2.295,   0.817,
        -0.821,  -2.299,  -3.777,  -5.255,
        -6.733,  -8.210,  -9.688, -11.166,
       -12.644, -14.122, -15.600, -17.077,
       -18.555, -19.554, -20.602, -21.271
        """, sep="," )*1000/20000

rc = np.fromstring("""
        3.843,   6.509,   9.185,  11.053,
       13.210,  14.924,  16.325,  17.495,
       18.465,  19.282,  19.951,  20.502,
       20.930,  21.240,  21.450,  21.559,
       21.559,  21.450,  21.240,  20.930,
       20.502,  19.951,  19.282,  18.465,
       17.495,  16.325,  14.924,  13.210,
       11.053,   9.185,   6.509,   3.843
       """, sep="," )*1000/20000

# Generate the flight path data
flight_paths = []

origin = np.array([0,0,0])


zdelta = 1000./20000

slanted = "SLANTED" in os.environ

if slanted == False:
    rr = rs
    zz = zs
    R = np.eye(3)
else:
    rr = rc
    zz = zc
    R = get_tilt_rotation()
pass

assert len(zz) == len(rr)
num = len(zz)

do_offset_after_tilt = False


#n_points_per_ring = 36
n_points_per_ring = 24
#n_points_per_ring = 12

for i in range(num):
    # Generate angles for this ring
    angles = np.linspace(0, 2*np.pi, n_points_per_ring, endpoint=False)

    for angle in angles:
        axis_dir = np.array([0,0,1])
        axis_point = np.array([0,0,zz[i]])
        eye_untilted = np.array([ rr[i] * np.cos(angle), rr[i] * np.sin(angle), zz[i]])
        tangent_untilted = np.array([-rr[i] * np.sin(angle),rr[i] * np.cos(angle),  0])
        tangent_untilted = tangent_untilted / np.linalg.norm(tangent_untilted)

        look_untilted = eye_untilted + tangent_untilted/20.  # look point ahead along tangent 

        radial_untilted = eye_untilted - axis_point
        radial_untilted = radial_untilted /  np.linalg.norm(radial_untilted)
        up_untilted = radial_untilted

        if do_offset_after_tilt == False: # offset for better visiability, by not being ontop of what want to see
            eye_untilted += up_untilted/20. + axis_dir/20.
            look_untilted += up_untilted/20.
        pass 

 
        eye = eye_untilted @ R
        look = look_untilted @ R
        up = up_untilted @ R


        if do_offset_after_tilt == True:
            eye_look_delta = up/20.    # radial offset, so can see what are targetting
            eye += eye_look_delta
            look += eye_look_delta
        pass


        # Create the 4x4 block for this position
        block = np.array([
            [eye[0], eye[1], eye[2], 1],  # Eye position
            [look[0], look[1], look[2], 1],  # Look position
            [up[0], up[1], up[2], 0],  # Up direction
            [0, 0, 0, 0]  # Padding row
        ])

        flight_paths.append(block)

# Convert to single numpy array
flight_paths = np.array(flight_paths)
fp = flight_paths.astype(np.float32)


print(f"Flight path data shape: {flight_paths.shape}")
print(f"Number of positions: {len(flight_paths)}")
print(f"Each block is 4x4: {flight_paths[0].shape}")
print("\nFirst block:")
print(flight_paths[0])

# Example: access specific elements
print("\nFirst eye position:", flight_paths[0][0, :3])
print("First look position:", flight_paths[0][1, :3])
print("First up direction:", flight_paths[0][2, :3])

if slanted:
    outname = "tilted_rings_2.npy"
else:
    outname = "tilted_rings_2_horizontals.npy"
pass

outpath = os.path.join(os.environ.get("OUTDIR","/tmp"),outname)
np.save(outpath, fp)

print(f"saved to {outpath} example commandline below")
cmdline=f"FULLSCREEN=0 VIEW={outpath} VIEWSLICE=[:] SGLM_InterpolatedView__STEPS=1000 cxr_min.sh"
print(cmdline)



