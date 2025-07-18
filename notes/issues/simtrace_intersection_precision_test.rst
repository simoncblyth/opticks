simtrace_intersection_precision_test
======================================

Overview
---------

simtrace is more appropriate for testing intersect precision
than input photons : so dont have to fight the simulation
randomization.

Hence, need to add some simtrace functionality
that uses SRecord::getPhotonAtTime

Need to do something analogous to input photon gensteps but for simtrace


cxt_precision.sh
-------------------

Without refine. Same simtrace intersect from 20m down to 0.4m ray by ray::

    In [13]: i=0 ; np.c_[s[i,:,1,:3],s[i,:,0,3], np.linalg.norm(s[i,:,1,:3]-s[i,-1,1,:3],axis=1)  ]
    Out[13]:
    array([[-1251.225, 20623.73 ,   830.09 , 20187.092,     0.497],
           [-1251.325, 20623.73 ,   829.648, 18388.34 ,     0.044],
           [-1251.125, 20623.73 ,   830.528, 16588.232,     0.946],
           [-1251.325, 20623.73 ,   829.648, 14789.932,     0.044],
           [-1251.377, 20623.73 ,   829.422, 12990.958,     0.187],
           [-1251.301, 20623.73 ,   829.757, 11191.409,     0.156],
           [-1251.326, 20623.73 ,   829.647,  9392.318,     0.043],
           [-1251.351, 20623.73 ,   829.535,  7593.228,     0.072],
           [-1251.339, 20623.73 ,   829.591,  5793.966,     0.014],
           [-1251.332, 20623.73 ,   829.619,  3994.733,     0.015],
           [-1251.334, 20623.73 ,   829.612,  2195.535,     0.007],
           [-1251.336, 20623.73 ,   829.605,   396.339,     0.   ]], dtype=float32)

    In [14]: i=1 ; np.c_[s[i,:,1,:3],s[i,:,0,3], np.linalg.norm(s[i,:,1,:3]-s[i,-1,1,:3],axis=1)  ]
    Out[14]:
    array([[-1248.073, 20623.73 ,   838.069, 20178.611,     0.32 ],
           [-1248.155, 20623.73 ,   837.711, 18379.773,     0.048],
           [-1248.154, 20623.73 ,   837.712, 16580.568,     0.046],
           [-1248.155, 20623.73 ,   837.711, 14781.364,     0.047],
           [-1248.196, 20623.73 ,   837.531, 12982.346,     0.232],
           [-1248.114, 20623.73 ,   837.89 , 11182.771,     0.137],
           [-1248.155, 20623.73 ,   837.711,  9383.751,     0.047],
           [-1248.144, 20623.73 ,   837.756,  7584.5  ,     0.001],
           [-1248.144, 20623.73 ,   837.756,  5785.295,     0.002],
           [-1248.142, 20623.73 ,   837.767,  3986.079,     0.01 ],
           [-1248.143, 20623.73 ,   837.761,  2186.881,     0.004],
           [-1248.144, 20623.73 ,   837.757,   387.681,     0.   ]], dtype=float32)

    In [15]: i=2 ; np.c_[s[i,:,1,:3],s[i,:,0,3], np.linalg.norm(s[i,:,1,:3]-s[i,-1,1,:3],axis=1)  ]
    Out[15]:
    array([[-1245.165, 20623.73 ,   844.988, 20171.22 ,     0.559],
           [-1245.305, 20623.73 ,   844.374, 18372.646,     0.071],
           [-1245.165, 20623.73 ,   844.987, 16572.812,     0.557],
           [-1245.339, 20623.73 ,   844.222, 14774.392,     0.227],
           [-1245.304, 20623.73 ,   844.376, 12975.029,     0.069],
           [-1245.252, 20623.73 ,   844.606, 11175.59 ,     0.167],
           [-1245.287, 20623.73 ,   844.454,  9376.54 ,     0.011],
           [-1245.296, 20623.73 ,   844.414,  7577.376,     0.03 ],
           [-1245.287, 20623.73 ,   844.452,  5778.133,     0.009],
           [-1245.289, 20623.73 ,   844.443,  3978.938,     0.   ],
           [-1245.287, 20623.73 ,   844.453,  2179.724,     0.01 ],
           [-1245.289, 20623.73 ,   844.443,   380.528,     0.   ]], dtype=float32)

    In [16]: i=3 ; np.c_[s[i,:,1,:3],s[i,:,0,3], np.linalg.norm(s[i,:,1,:3]-s[i,-1,1,:3],axis=1)  ]
    Out[16]:
    array([[-1242.59 , 20623.73 ,   850.446, 20165.326,     0.219],
           [-1242.528, 20623.73 ,   850.718, 18365.844,     0.498],
           [-1242.715, 20623.73 ,   849.898, 16567.479,     0.343],
           [-1242.621, 20623.73 ,   850.307, 14767.854,     0.077],
           [-1242.652, 20623.73 ,   850.172, 12968.789,     0.062],
           [-1242.621, 20623.73 ,   850.308, 11169.444,     0.078],
           [-1242.621, 20623.73 ,   850.307,  9370.24 ,     0.077],
           [-1242.652, 20623.73 ,   850.172,  7571.175,     0.062],
           [-1242.645, 20623.73 ,   850.206,  5771.935,     0.027],
           [-1242.639, 20623.73 ,   850.232,  3972.704,     0.001],
           [-1242.638, 20623.73 ,   850.235,  2173.496,     0.003],
           [-1242.638, 20623.73 ,   850.232,   374.293,     0.   ]], dtype=float32)

    In [17]: i=4 ; np.c_[s[i,:,1,:3],s[i,:,0,3], np.linalg.norm(s[i,:,1,:3]-s[i,-1,1,:3],axis=1)  ]
    Out[17]:
    array([[-1240.084, 20623.73 ,   855.607, 20159.738,     0.228],
           [-1240.083, 20623.73 ,   855.608, 18360.533,     0.229],
           [-1240.14 , 20623.73 ,   855.359, 16561.584,     0.026],
           [-1240.168, 20623.73 ,   855.235, 14762.506,     0.154],
           [-1240.14 , 20623.73 ,   855.361, 12963.173,     0.025],
           [-1240.14 , 20623.73 ,   855.36 , 11163.969,     0.026],
           [-1240.14 , 20623.73 ,   855.36 ,  9364.764,     0.026],
           [-1240.133, 20623.73 ,   855.391,  7565.528,     0.006],
           [-1240.14 , 20623.73 ,   855.359,  5766.356,     0.026],
           [-1240.133, 20623.73 ,   855.391,  3967.117,     0.006],
           [-1240.134, 20623.73 ,   855.387,  2167.917,     0.002],
           [-1240.134, 20623.73 ,   855.385,   368.714,     0.   ]], dtype=float32)


Delta for all photon::


    In [21]: s[:,:,1,:3] - s[:,-1,1,:3][:,np.newaxis]
    Out[21]:
    array([[[ 0.111,  0.   ,  0.485],
            [ 0.01 ,  0.   ,  0.043],
            [ 0.211,  0.   ,  0.923],
            [ 0.01 ,  0.   ,  0.043],
            [-0.041,  0.   , -0.183],
            ...,
            [-0.016,  0.   , -0.07 ],
            [-0.003,  0.   , -0.014],
            [ 0.004,  0.   ,  0.014],
            [ 0.002,  0.   ,  0.007],
            [ 0.   ,  0.   ,  0.   ]],

          [[ 0.228,  0.   ,  1.132],
            [ 0.034,  0.   ,  0.168],
            [ 0.132,  0.   ,  0.654],
            [-0.015,  0.   , -0.077],
            [-0.016,  0.   , -0.079],
            ...,
            [ 0.009,  0.   ,  0.045],
            [ 0.009,  0.   ,  0.045],
            [-0.   ,  0.   , -0.001],
            [-0.001,  0.   , -0.008],
            [ 0.   ,  0.   ,  0.   ]]], shape=(155, 12, 3), dtype=float32)

    In [22]:


Convert the delta vectors into distances::

    In [35]: np.set_printoptions(linewidth=200,edgeitems=20)

    In [36]: e = np.linalg.norm((s[:,:,1,:3] - s[:,-1,1,:3][:,np.newaxis]).reshape(-1,3),axis=1).reshape(-1,12) ; e
    Out[36]:
    array([[0.497, 0.044, 0.946, 0.044, 0.187, 0.156, 0.043, 0.072, 0.014, 0.015, 0.007, 0.   ],
           [0.32 , 0.048, 0.046, 0.047, 0.232, 0.137, 0.047, 0.001, 0.002, 0.01 , 0.004, 0.   ],
           [0.559, 0.071, 0.557, 0.227, 0.069, 0.167, 0.011, 0.03 , 0.009, 0.   , 0.01 , 0.   ],
           [0.219, 0.498, 0.343, 0.077, 0.062, 0.078, 0.077, 0.062, 0.027, 0.001, 0.003, 0.   ],
           [0.228, 0.229, 0.026, 0.154, 0.025, 0.026, 0.026, 0.006, 0.026, 0.006, 0.002, 0.   ],
           [0.263, 0.209, 0.028, 0.093, 0.027, 0.033, 0.028, 0.002, 0.003, 0.004, 0.004, 0.   ],
           [0.097, 0.095, 0.095, 0.124, 0.013, 0.069, 0.041, 0.014, 0.014, 0.007, 0.007, 0.   ],
           [0.075, 0.119, 0.271, 0.025, 0.024, 0.027, 0.022, 0.026, 0.002, 0.004, 0.004, 0.   ],
           [0.094, 0.096, 0.264, 0.072, 0.013, 0.054, 0.054, 0.029, 0.009, 0.002, 0.001, 0.   ],
           [0.067, 0.068, 0.07 , 0.007, 0.007, 0.067, 0.007, 0.007, 0.007, 0.002, 0.   , 0.   ],
           [0.269, 0.001, 0.133, 0.002, 0.002, 0.032, 0.003, 0.016, 0.001, 0.002, 0.002, 0.   ],
           [0.12 , 0.139, 0.01 , 0.055, 0.076, 0.01 , 0.023, 0.026, 0.01 , 0.006, 0.   , 0.   ],
           [0.248, 0.007, 0.007, 0.073, 0.073, 0.023, 0.024, 0.007, 0.008, 0.008, 0.006, 0.   ],
           [0.184, 0.067, 0.051, 0.007, 0.05 , 0.007, 0.037, 0.021, 0.007, 0.   , 0.001, 0.   ],
           [0.138, 0.093, 0.024, 0.094, 0.034, 0.051, 0.005, 0.019, 0.008, 0.002, 0.002, 0.   ],
           [0.109, 0.109, 0.003, 0.003, 0.003, 0.062, 0.025, 0.004, 0.01 , 0.004, 0.001, 0.   ],
           [0.091, 0.093, 0.09 , 0.033, 0.034, 0.062, 0.005, 0.006, 0.006, 0.001, 0.001, 0.   ],
           [0.197, 0.024, 0.195, 0.03 , 0.03 , 0.03 , 0.025, 0.012, 0.012, 0.005, 0.002, 0.   ],
           [0.109, 0.109, 0.108, 0.001, 0.   , 0.001, 0.001, 0.014, 0.001, 0.   , 0.   , 0.   ],
           [0.176, 0.033, 0.07 , 0.033, 0.034, 0.033, 0.019, 0.007, 0.005, 0.001, 0.002, 0.   ],
           ...,
           [0.052, 0.161, 0.053, 0.001, 0.053, 0.   , 0.026, 0.   , 0.013, 0.   , 0.002, 0.   ],
           [0.136, 0.027, 0.079, 0.027, 0.027, 0.001, 0.026, 0.   , 0.014, 0.003, 0.   , 0.   ],
           [0.123, 0.123, 0.124, 0.042, 0.012, 0.066, 0.015, 0.016, 0.015, 0.002, 0.002, 0.   ],
           [0.133, 0.01 , 0.011, 0.053, 0.01 , 0.021, 0.01 , 0.006, 0.006, 0.006, 0.004, 0.   ],
           [0.222, 0.098, 0.025, 0.027, 0.026, 0.005, 0.035, 0.011, 0.011, 0.004, 0.003, 0.   ],
           [0.205, 0.078, 0.048, 0.049, 0.049, 0.018, 0.018, 0.034, 0.003, 0.007, 0.003, 0.   ],
           [0.066, 0.066, 0.196, 0.001, 0.002, 0.034, 0.002, 0.017, 0.001, 0.002, 0.002, 0.   ],
           [0.068, 0.203, 0.07 , 0.004, 0.003, 0.005, 0.101, 0.02 , 0.003, 0.004, 0.002, 0.   ],
           [0.121, 0.016, 0.015, 0.085, 0.053, 0.019, 0.054, 0.019, 0.002, 0.006, 0.004, 0.   ],
           [0.173, 0.031, 0.173, 0.038, 0.039, 0.002, 0.004, 0.003, 0.003, 0.001, 0.001, 0.   ],
           [0.095, 0.099, 0.241, 0.049, 0.048, 0.025, 0.012, 0.012, 0.006, 0.003, 0.004, 0.   ],
           [0.183, 0.033, 0.182, 0.043, 0.032, 0.004, 0.032, 0.005, 0.014, 0.009, 0.002, 0.   ],
           [0.101, 0.055, 0.056, 0.136, 0.136, 0.018, 0.06 , 0.017, 0.002, 0.002, 0.002, 0.   ],
           [0.094, 0.08 , 0.255, 0.082, 0.083, 0.05 , 0.049, 0.016, 0.015, 0.001, 0.002, 0.   ],
           [0.312, 0.049, 0.051, 0.051, 0.04 , 0.04 , 0.041, 0.017, 0.005, 0.001, 0.003, 0.   ],
           [0.195, 0.196, 0.194, 0.004, 0.003, 0.004, 0.052, 0.043, 0.02 , 0.004, 0.002, 0.   ],
           [0.08 , 0.082, 0.117, 0.218, 0.018, 0.031, 0.018, 0.019, 0.043, 0.006, 0.   , 0.   ],
           [0.05 , 0.191, 0.29 , 0.052, 0.07 , 0.009, 0.051, 0.01 , 0.01 , 0.006, 0.006, 0.   ],
           [0.28 , 0.3  , 0.277, 0.133, 0.011, 0.083, 0.134, 0.011, 0.011, 0.002, 0.002, 0.   ],
           [1.155, 0.172, 0.667, 0.079, 0.081, 0.08 , 0.171, 0.046, 0.046, 0.001, 0.008, 0.   ]], shape=(155, 12), dtype=float32)


Show the shooting distance::

    In [37]: d = s[:,:,0,3] ; d
    Out[37]:
    array([[20187.092, 18388.34 , 16588.232, 14789.932, 12990.958, 11191.409,  9392.318,  7593.228,  5793.966,  3994.733,  2195.535,   396.339],
           [20178.611, 18379.773, 16580.568, 14781.364, 12982.346, 11182.771,  9383.751,  7584.5  ,  5785.295,  3986.079,  2186.881,   387.681],
           [20171.22 , 18372.646, 16572.812, 14774.392, 12975.029, 11175.59 ,  9376.54 ,  7577.376,  5778.133,  3978.938,  2179.724,   380.528],
           [20165.326, 18365.844, 16567.479, 14767.854, 12968.789, 11169.444,  9370.24 ,  7571.175,  5771.935,  3972.704,  2173.496,   374.293],
           [20159.738, 18360.533, 16561.584, 14762.506, 12963.173, 11163.969,  9364.764,  7565.528,  5766.356,  3967.117,  2167.917,   368.714],
           [20154.605, 18355.873, 16556.432, 14757.348, 12958.023, 11158.879,  9359.614,  7560.439,  5761.236,  3962.024,  2162.818,   363.62 ],
           [20150.06 , 18350.857, 16551.652, 14752.669, 12953.353, 11154.204,  9354.89 ,  7555.739,  5756.507,  3957.323,  2158.105,   358.907],
           [20141.54 , 18342.527, 16542.934, 14744.024, 12944.819, 11145.563,  9346.408,  7547.155,  5747.975,  3948.777,  2149.562,   350.363],
           [20126.93 , 18327.723, 16528.35 , 14729.481, 12930.192, 11130.946,  9331.742,  7532.62 ,  5733.394,  3934.178,  2134.977,   335.771],
           [20114.582, 18315.377, 16516.17 , 14717.043, 12917.838, 11118.56 ,  9319.429,  7520.223,  5721.019,  3921.811,  2122.604,   323.397],
           [20103.52 , 18304.584, 16505.512, 14706.173, 12906.969, 11107.798,  9308.558,  7509.373,  5710.15 ,  3910.95 ,  2111.745,   312.538],
           [20101.143, 18302.197, 16502.863, 14703.594, 12904.521, 11105.25 ,  9306.012,  7506.856,  5707.636,  3908.415,  2109.217,   310.012],
           [20098.555, 18299.605, 16500.4  , 14701.263, 12902.058, 11102.757,  9303.552,  7504.364,  5705.175,  3905.954,  2106.751,   307.554],
           [20087.174, 18288.086, 16488.998, 14689.737, 12890.589, 11091.327,  9292.093,  7492.946,  5693.713,  3894.516,  2095.31 ,   296.107],
           [20085.082, 18286.11 , 16486.787, 14687.7  , 12888.437, 11089.146,  9289.997,  7490.808,  5691.576,  3892.377,  2093.175,   293.97 ],
           [20083.023, 18283.818, 16484.727, 14685.521, 12886.317, 11087.171,  9287.88 ,  7488.703,  5689.485,  3890.294,  2091.086,   291.881],
           [20080.99 , 18281.783, 16482.582, 14683.434, 12884.229, 11084.996,  9285.848,  7486.642,  5687.438,  3888.24 ,  2089.033,   289.83 ],
           [20078.879, 18279.895, 16480.473, 14681.433, 12882.228, 11083.022,  9283.873,  7484.655,  5685.451,  3886.229,  2087.026,   287.825],
           [20075.07 , 18276.084, 16476.662, 14677.563, 12878.36 , 11079.155,  9279.95 ,  7480.76 ,  5681.542,  3882.336,  2083.133,   283.927],
           [20071.254, 18272.258, 16472.951, 14673.85 , 12874.645, 11075.439,  9276.184,  7477.005,  5677.788,  3878.588,  2079.381,   280.178],
           ...,
           [20074.371, 18275.057, 16476.066, 14676.811, 12877.658, 11078.4  ,  9279.222,  7479.991,  5680.773,  3881.581,  2082.375,   283.172],
           [20076.195, 18277.1  , 16478.   , 14678.69 , 12879.486, 11080.307,  9281.076,  7481.898,  5682.68 ,  3883.491,  2084.283,   285.079],
           [20078.152, 18278.947, 16479.742, 14680.703, 12881.446, 11082.186,  9283.062,  7483.858,  5684.652,  3885.434,  2086.227,   287.024],
           [20092.928, 18293.846, 16494.64 , 14695.499, 12896.233, 11097.059,  9297.823,  7498.634,  5699.429,  3900.213,  2101.01 ,   301.809],
           [20095.143, 18296.062, 16496.98 , 14697.778, 12898.573, 11099.337,  9300.103,  7500.942,  5701.738,  3902.52 ,  2103.316,   304.114],
           [20097.521, 18298.443, 16499.365, 14700.161, 12900.957, 11101.721,  9302.516,  7503.328,  5704.092,  3904.89 ,  2105.676,   306.474],
           [20100.082, 18300.877, 16501.543, 14702.533, 12903.328, 11104.091,  9304.918,  7505.699,  5706.508,  3907.303,  2108.1  ,   308.898],
           [20102.562, 18303.223, 16504.152, 14705.013, 12905.81 , 11106.603,  9307.302,  7508.178,  5708.99 ,  3909.793,  2110.582,   311.38 ],
           [20107.682, 18308.613, 16509.408, 14710.273, 12910.932, 11111.761,  9312.521,  7513.351,  5714.164,  3914.954,  2115.753,   316.552],
           [20110.328, 18311.266, 16511.92 , 14712.925, 12913.723, 11114.48 ,  9315.277,  7516.072,  5716.868,  3917.659,  2118.457,   319.251],
           [20113.188, 18313.98 , 16514.633, 14715.717, 12916.513, 11117.234,  9318.066,  7518.862,  5719.641,  3920.445,  2121.232,   322.033],
           [20115.969, 18316.914, 16517.56 , 14718.579, 12919.301, 11120.132,  9320.891,  7521.723,  5722.5  ,  3923.3  ,  2124.101,   324.899],
           [20122.084, 18323.035, 16523.832, 14724.707, 12925.503, 11126.18 ,  9326.896,  7527.769,  5728.545,  3929.34 ,  2130.141,   330.934],
           [20132.03 , 18333.   , 16533.969, 14734.593, 12935.388, 11136.051,  9336.848,  7537.708,  5738.502,  3939.283,  2140.081,   340.873],
           [20135.416, 18336.572, 16537.37 , 14738.165, 12938.869, 11139.664,  9340.459,  7541.278,  5742.096,  3942.885,  2143.678,   344.478],
           [20139.314, 18340.11 , 16540.906, 14741.892, 12942.688, 11143.481,  9344.229,  7545.12 ,  5745.892,  3946.663,  2147.464,   348.258],
           [20143.41 , 18344.203, 16545.197, 14746.094, 12946.689, 11147.436,  9348.28 ,  7549.076,  5749.896,  3950.641,  2151.443,   352.239],
           [20156.965, 18358.002, 16558.316, 14759.349, 12960.267, 11161.001,  9361.736,  7562.592,  5763.388,  3964.168,  2164.964,   365.765],
           [20167.727, 18369.102, 16569.32 , 14770.259, 12971.198, 11172.066,  9372.644,  7573.584,  5774.38 ,  3975.166,  2175.961,   376.754],
           [20190.143, 18391.922, 16592.22 , 14793.763, 12994.56 , 11195.354,  9395.898,  7596.819,  5797.615,  3998.457,  2199.26 ,   400.047]], shape=(155, 12), dtype=float32)
















Most precise at end very close to 2nd BT::

    In [3]: r.f.record[rw[0],:5,0]
    Out[3]:
    array([[  3234.466,  20623.73 ,  20512.5  ,      0.1  ],
           [ -1251.225,  20623.73 ,    830.09 ,     89.86 ],
           [ -1251.335,  20623.73 ,    829.608,     89.862],
           [ -1266.823,  20623.73 ,    761.648,     90.172],
           [ -6374.514,  20623.73 , -21650.   ,    192.379]], dtype=float32)

    In [4]: r.f.record[rw[1],:5,0]
    Out[4]:
    array([[  3234.497,  20623.73 ,  20512.492,      0.1  ],
           [ -1248.073,  20623.73 ,    838.069,     89.822],
           [ -1248.144,  20623.73 ,    837.756,     89.824],
           [ -1267.481,  20623.73 ,    752.885,     90.211],
           [ -6371.695,  20623.73 , -21650.   ,    192.376]], dtype=float32)

    In [5]: s[1,:,1]
    Out[5]:
    array([[-1248.073, 20623.73 ,   838.069,     0.05 ],
           [-1248.155, 20623.73 ,   837.711,     0.05 ],
           [-1248.154, 20623.73 ,   837.712,     0.05 ],
           [-1248.155, 20623.73 ,   837.711,     0.05 ],
           [-1248.196, 20623.73 ,   837.531,     0.05 ],
           [-1248.114, 20623.73 ,   837.89 ,     0.05 ],
           [-1248.155, 20623.73 ,   837.711,     0.05 ],
           [-1248.144, 20623.73 ,   837.756,     0.05 ],
           [-1248.144, 20623.73 ,   837.756,     0.05 ],
           [-1248.142, 20623.73 ,   837.767,     0.05 ],
           [-1248.143, 20623.73 ,   837.761,     0.05 ],
           [-1248.144, 20623.73 ,   837.757,     0.05 ]], dtype=float32)




Changes
--------

* add OpticksGenstep_INPUT_PHOTON_SIMTRACE
* generalize qsim::generate_photon_simtrace
* add simtrace layout to SRecord::getPhotonAtTime and SRecord::getSimtraceAtTime


::

    2447 inline QSIM_METHOD void qsim::generate_photon_simtrace(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2448 {
    2449     const int& gencode = gs.q0.i.x ;
    2450     switch(gencode)
    2451     {
    2452         case OpticksGenstep_FRAME:                   generate_photon_simtrace_frame(p, rng, gs, photon_id, genstep_id ); break ;
    2453         case OpticksGenstep_INPUT_PHOTON_SIMTRACE:   { p = (quad4&)evt->photon[photon_id] ; }                          ; break ;
    2454     }
    2455 }
    2456
    2457 inline QSIM_METHOD void qsim::generate_photon_simtrace_frame(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2458 {
    2459     C4U gsid ;
    2460
    2461     //int gencode          = gs.q0.i.x ;
    2462     int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }



TOFIX : more than one simtrace layout
-----------------------------------------

::

    469 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    470 {
    471     unsigned idx = launch_idx.x ;
    472     sevent* evt  = params.evt ;
    473     if (idx >= evt->num_simtrace) return;    // num_slot for multi launch simtrace ?
    474
    475     unsigned genstep_idx = evt->seed[idx] ;
    476     unsigned photon_idx  = params.photon_slot_offset + idx ;
    477     // photon_idx same as idx for first launch, offset beyond first for multi-launch
    478
    479 #if defined(DEBUG_PIDX)
    480     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d photon_idx %d  genstep_idx %d evt->num_simtrace %d \n", idx, photon_idx, genstep_idx, evt->num_simtrace );
    481 #endif
    482
    483     const quad6& gs = evt->genstep[genstep_idx] ;
    484
    485     qsim* sim = params.sim ;
    486     RNG rng ;
    487     sim->rng->init(rng, 0, photon_idx) ;
    488
    489     quad4 p ;
    490     sim->generate_photon_simtrace(p, rng, gs, photon_idx, genstep_idx );
    491
    492
    493     // HUH: this is not the layout of sevent::add_simtrace
    494     const float3& pos = (const float3&)p.q0.f  ;
    495     const float3& mom = (const float3&)p.q1.f ;
    496





Review
---------

input photon gensteps
~~~~~~~~~~~~~~~~~~~~~~~

::

    094 NP* SEvent::MakeInputPhotonGenstep(const NP* input_photon, const sframe& fr )
     95 {
     96     std::vector<quad6> qgs(1) ;
     97     qgs[0].zero() ;
     98     qgs[0] = MakeInputPhotonGenstep_(input_photon, fr );
     99     NP* ipgs = NPX::ArrayFromVec<float,quad6>( qgs, 6, 4) ;
    100     return ipgs ;
    101 }

    117 quad6 SEvent::MakeInputPhotonGenstep_(const NP* input_photon, const sframe& fr )
    118 {
    119     LOG(LEVEL) << " input_photon " << NP::Brief(input_photon) ;
    120
    121     quad6 ipgs ;
    122     ipgs.zero();
    123     ipgs.set_gentype( OpticksGenstep_INPUT_PHOTON );
    124     ipgs.set_numphoton(  input_photon->shape[0]  );
    125     fr.m2w.write(ipgs); // copy fr.m2w into ipgs.q2,q3,q4,q5
    126     return ipgs ;
    127 }



    0317 int QEvent::setGenstepUpload(const quad6* qq0, int gs_start, int gs_stop )
     318 {
     ...
     395     int gencode0 = SGenstep::GetGencode(qq, 0) ; // gencode of first genstep or OpticksGenstep_INVALID for qq nullptr
     396
     397     if(OpticksGenstep_::IsFrame(gencode0))   // OpticksGenstep_FRAME  (HMM: Obtuse, maybe change to SIMTRACE ?)
     398     {
     399         setNumSimtrace( evt->num_seed );
     400     }
     401     else if(OpticksGenstep_::IsInputPhoton(gencode0)) // OpticksGenstep_INPUT_PHOTON  (NOT: _TORCH)
     402     {
     403         setInputPhotonAndUpload();
     404     }
     405     else
     406     {
     407         setNumPhoton( evt->num_seed );  // *HEAVY* : photon, rec, record may be allocated here depending on SEventConfig
     408     }
     409     upload_count += 1 ;



     497 void QEvent::setInputPhotonAndUpload()
     498 {
     499     LOG_IF(info, LIFECYCLE) ;
     500     LOG(LEVEL);
     501     input_photon = sev->gatherInputPhoton();
     502     checkInputPhoton();
     503
     504     int numph = input_photon->shape[0] ;
     505     setNumPhoton( numph );
     506     QU::copy_host_to_device<sphoton>( evt->photon, (sphoton*)input_photon->bytes(), numph );
     507 }


qsim::generate_photon::


    2509 inline QSIM_METHOD void qsim::generate_photon(sphoton& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2510 {
    2511     const int& gencode = gs.q0.i.x ;
    2512     switch(gencode)
    2513     {
    2514         case OpticksGenstep_CARRIER:         scarrier::generate(     p, rng, gs, photon_id, genstep_id)  ; break ;
    2515         case OpticksGenstep_TORCH:           storch::generate(       p, rng, gs, photon_id, genstep_id ) ; break ;
    2516
    2517         case OpticksGenstep_G4Cerenkov_modified:
    2518         case OpticksGenstep_CERENKOV:
    2519                                               cerenkov->generate(    p, rng, gs, photon_id, genstep_id ) ; break ;
    2520
    2521         case OpticksGenstep_DsG4Scintillation_r4695:
    2522         case OpticksGenstep_SCINTILLATION:
    2523                                               scint->generate(        p, rng, gs, photon_id, genstep_id ) ; break ;
    2524
    2525         case OpticksGenstep_INPUT_PHOTON:    { p = evt->photon[photon_id] ; p.set_flag(TORCH) ; }        ; break ;
    2526         default:                             generate_photon_dummy(  p, rng, gs, photon_id, genstep_id)  ; break ;
    2527     }
    2528     p.set_idx(photon_id);
    2529 }




cxt_min.sh configures simtrace gensteps with CEGS CEHIGH
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    188 ## see SFrameGenstep::StandardizeCEGS for CEGS/CEHIGH [4]/[7]/[8] layouts
    189
    190 export CEGS=16:0:9:2000   # [4] XZ default
    191 #export CEGS=16:0:9:1000  # [4] XZ default
    192 #export CEGS=16:0:9:100   # [4] XZ reduce rays for faster rsync
    193 #export CEGS=16:9:0:1000  # [4] try XY
    194
    195 export CEHIGH_0=-16:16:0:0:-4:4:2000:4
    196 export CEHIGH_1=-16:16:0:0:4:8:2000:4
    197
    198 #export CEHIGH_0=16:0:9:0:0:10:2000     ## [7] dz:10 aim to land another XZ grid above in Z 16:0:9:2000
    199 #export CEHIGH_1=-4:4:0:0:-9:9:2000:5   ## [8]
    200 #export CEHIGH_2=-4:4:0:0:10:28:2000:5  ## [8]
    201



QSim::simtrace
~~~~~~~~~~~~~~~~

::

     664 double QSim::simtrace(int eventID)
     665 {
     666     sev->beginOfEvent(eventID);
     667
     668     NP* igs = sev->makeGenstepArrayFromVector();
     669     int rc = event->setGenstepUpload_NP(igs) ;
     670
     671     LOG_IF(error, rc != 0) << " QEvent::setGenstep ERROR : no gensteps collected : will skip cx.simtrace " ;
     672
     673     sev->t_PreLaunch = sstamp::Now() ;
     674     double dt = rc == 0 && cx != nullptr ? cx->simtrace_launch() : -1. ;
     675     sev->t_PostLaunch = sstamp::Now() ;
     676     sev->t_Launch = dt ;
     677
     678     // see ~/o/notes/issues/cxt_min_simtrace_revival.rst
     679     sev->gather();
     680
     681     sev->topfold->concat();
     682     sev->topfold->clear_subfold();
     683
     684     sev->endOfEvent(eventID);
     685
     686     return dt ;
     687 }




SEvt::addInputGenstep
~~~~~~~~~~~~~~~~~~~~~~~

::

     859 void SEvt::addInputGenstep()
     860 {
     861     LOG_IF(info, LIFECYCLE) << id() ;
     862     LOG(LEVEL);
     863
     864     if(SEventConfig::IsRGModeSimtrace())
     865     {
     866         const char* frs = frame.get_frs() ; // nullptr when default -1 : meaning all geometry
     867
     868         LOG_IF(info, SIMTRACE )
     869             << "[" << SEvt__SIMTRACE << "] "
     870             << " frame.get_frs " << ( frs ? frs : "-" ) ;
     871             ;
     872
     873         //if(frs) SEventConfig::SetEventReldir(frs); // dont do that, default is more standard
     874         // doing this is hangover from separate simtracing of related volumes presumably
     875
     876         NP* gs = SFrameGenstep::MakeCenterExtentGenstep_FromFrame(frame);
     877         LOG_IF(info, SIMTRACE)
     878             << "[" << SEvt__SIMTRACE << "] "
     879             << " simtrace gs " << ( gs ? gs->sstr() : "-" )
     880             ;
     881
     882         addGenstep(gs);
     883
     884         if(frame.is_hostside_simtrace()) setFrame_HostsideSimtrace();
     885     }


CSGOptiX7.cu::

    469 static __forceinline__ __device__ void simtrace( const uint3& launch_idx, const uint3& dim, quad2* prd )
    470 {
    471     unsigned idx = launch_idx.x ;
    472     sevent* evt  = params.evt ;
    473     if (idx >= evt->num_simtrace) return;    // num_slot for multi launch simtrace ?
    474
    475     unsigned genstep_idx = evt->seed[idx] ;
    476     unsigned photon_idx  = params.photon_slot_offset + idx ;
    477     // photon_idx same as idx for first launch, offset beyond first for multi-launch
    478
    479 #if defined(DEBUG_PIDX)
    480     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d photon_idx %d  genstep_idx %d evt->num_simtrace %d \n", idx, photon_idx, genstep_idx, evt->num_simtrace );
    481 #endif
    482
    483     const quad6& gs = evt->genstep[genstep_idx] ;
    484
    485     qsim* sim = params.sim ;
    486     RNG rng ;
    487     sim->rng->init(rng, 0, photon_idx) ;
    488
    489     quad4 p ;
    490     sim->generate_photon_simtrace(p, rng, gs, photon_idx, genstep_idx );
    491
    492     const float3& pos = (const float3&)p.q0.f  ;
    493     const float3& mom = (const float3&)p.q1.f ;
    494
    495
    496 #if defined(DEBUG_PIDX)
    497     if(photon_idx == 0) printf("//CSGOptiX7.cu : simtrace idx %d pos.xyz %7.3f,%7.3f,%7.3f mom.xyz %7.3f,%7.3f,%7.3f  \n", idx, pos.x, pos.y, pos.z, mom.x, mom.y, mom.z );
    498 #endif
    499
    500
    501
    502
    503     trace<false>(
    504         params.handle,
    505         pos,
    506         mom,
    507         params.tmin,
    508         params.tmax,
    509         prd,
    510         params.vizmask,
    511         params.PropagateRefineDistance
    512     );
    513
    514     evt->add_simtrace( idx, p, prd, params.tmin );  // sevent
    515     // not photon_idx, needs to go from zero for photons from a slice of genstep array
    516 }



qsim::generate_photon_simtrace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2447 inline QSIM_METHOD void qsim::generate_photon_simtrace(quad4& p, RNG& rng, const quad6& gs, unsigned photon_id, unsigned genstep_id ) const
    2448 {
    2449     C4U gsid ;
    2450
    2451     //int gencode          = gs.q0.i.x ;
    2452     int gridaxes           = gs.q0.i.y ;  // { XYZ, YZ, XZ, XY }
    2453     gsid.u                 = gs.q0.i.z ;
    2454     //unsigned num_photons = gs.q0.u.w ;
    2455
    2456     p.q0.f.x = gs.q1.f.x ;   // start with genstep local frame position, typically origin  (0,0,0)
    2457     p.q0.f.y = gs.q1.f.y ;
    2458     p.q0.f.z = gs.q1.f.z ;
    2459     p.q0.f.w = 1.f ;
    2460
    2461     //printf("//qsim.generate_photon_simtrace gridaxes %d gs.q1 (%10.4f %10.4f %10.4f %10.4f) \n", gridaxes, gs.q1.f.x, gs.q1.f.y, gs.q1.f.z, gs.q1.f.w );
    2462
    2463     float u0 = curand_uniform(&rng);
    2464     float sinPhi, cosPhi;
    2465 #if defined(MOCK_CURAND) || defined(MOCK_CUDA)
    2466     __sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
    2467 #else
    2468     sincosf(2.f*M_PIf*u0,&sinPhi,&cosPhi);
    2469 #endif
    2470
    2471     float u1 = curand_uniform(&rng);
    2472     float cosTheta = 2.f*u1 - 1.f ;
    2473     float sinTheta = sqrtf(1.f-cosTheta*cosTheta) ;
    2474
    2475     //printf("//qsim.generate_photon_simtrace u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f \n", u0, sinPhi, cosPhi );
    2476     //printf("//qsim.generate_photon_simtrace u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n", u1, sinTheta, cosTheta );
    2477     //printf("//qsim.generate_photon_simtrace  u0 %10.4f sinPhi   %10.4f cosPhi   %10.4f u1 %10.4f sinTheta %10.4f cosTheta %10.4f \n",  u0, sinPhi, cosPhi, u1, sinTheta, cosTheta );
    2478
    2479     switch( gridaxes )
    2480     {
    2481         case YZ:  { p.q1.f.x = 0.f    ;  p.q1.f.y = cosPhi ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ;
    2482         case XZ:  { p.q1.f.x = cosPhi ;  p.q1.f.y = 0.f    ;  p.q1.f.z = sinPhi ;  p.q1.f.w = 0.f ; } ; break ;
    2483         case XY:  { p.q1.f.x = cosPhi ;  p.q1.f.y = sinPhi ;  p.q1.f.z = 0.f    ;  p.q1.f.w = 0.f ; } ; break ;
    2484         case XYZ: { p.q1.f.x = sinTheta*cosPhi ;
    2485                     p.q1.f.y = sinTheta*sinPhi ;
    2486                     p.q1.f.z = cosTheta        ;
    2487                     p.q1.f.w = 0.f ; } ; break ;   // previously used XZ
    2488     }
    2489
    2490
    2491     qat4 qt(gs) ; // copy 4x4 transform from last 4 quads of genstep
    2492     qt.right_multiply_inplace( p.q0.f, 1.f );   // position
    2493     qt.right_multiply_inplace( p.q1.f, 0.f );   // direction



