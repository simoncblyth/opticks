GMeshLib_uninitialized_transforms
===================================


Make some fresh geocaches into slot 101 and 103 with analytic enabled::

    epsilon:optickscore blyth$ OPTICKS_RESOURCE_LAYOUT=101 OKTest -G --gltf 1
    epsilon:optickscore blyth$ OPTICKS_RESOURCE_LAYOUT=103 OKTest -G --gltf 3


Using gltf 1 vs 3 does make a difference::

    epsilon:96ff965744a2f6b78c24e33c80d3a4cd blyth$ diff -r --brief 101 103
    Files 101/GMergedMeshAnalytic/0/bbox.npy and 103/GMergedMeshAnalytic/0/bbox.npy differ
    Files 101/GMergedMeshAnalytic/0/boundaries.npy and 103/GMergedMeshAnalytic/0/boundaries.npy differ
    Files 101/GMergedMeshAnalytic/0/center_extent.npy and 103/GMergedMeshAnalytic/0/center_extent.npy differ
    Files 101/GMergedMeshAnalytic/0/colors.npy and 103/GMergedMeshAnalytic/0/colors.npy differ
    ...
    Files 101/GMergedMeshAnalytic/5/sensors.npy and 103/GMergedMeshAnalytic/5/sensors.npy differ
    Files 101/GMergedMeshAnalytic/5/vertices.npy and 103/GMergedMeshAnalytic/5/vertices.npy differ

    Files 101/GMeshLib/24/transforms.npy and 103/GMeshLib/24/transforms.npy differ
    Files 101/GMeshLib/42/transforms.npy and 103/GMeshLib/42/transforms.npy differ
    epsilon:96ff965744a2f6b78c24e33c80d3a4cd blyth$ 


Not expecting difference in transforms, reveals a bug in meshes 24 and 42.
Looks like an uninitialized transform::

    n [1]: a = np.load("101/GMeshLib/24/transforms.npy") ; b = np.load("103/GMeshLib/24/transforms.npy")

    In [2]: a.shape
    Out[2]: (1, 16)

    In [3]: b.shape
    Out[3]: (1, 16)

    In [4]: a
    Out[4]: array([[ 0., -0.,  0., -0.,  0.,  0.,  0.,  0., nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

    In [5]: b
    Out[5]: array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)




    In [6]: a = np.load("101/GMeshLib/42/transforms.npy") ; b = np.load("103/GMeshLib/42/transforms.npy")

    In [7]: a.shape, b.shape
    Out[7]: ((1, 16), (1, 16))

    In [8]: a
    Out[8]: array([[ 0., -0.,  0., -0.,  0.,  0.,  0.,  0., nan,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)

    In [9]: b
    Out[9]: 
    array([[ 0.0000e+00,  4.6566e-10,  0.0000e+00,  4.6566e-10, -1.9025e-23,  4.5907e-41, -1.9331e-23,  4.5907e-41, -1.9035e-23,  4.5907e-41,  1.7000e+22,  3.2230e-44,  3.2230e-44,  4.6566e-10,
             0.0000e+00,  4.6566e-10]], dtype=float32)

    In [10]: 




