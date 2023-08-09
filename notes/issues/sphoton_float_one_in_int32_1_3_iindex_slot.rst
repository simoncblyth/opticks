sphoton_float_one_in_int32_1_3_iindex_slot
=============================================


::

    In [19]: np.ones(1, dtype=np.float32).view(np.int32)
    Out[19]: array([1065353216], dtype=int32)


    In [15]: np.where( a.f.record.view(np.int32) == 1065353216 )
    Out[15]: 
    (array([    0,     1,     2,     3,     4, ..., 99995, 99996, 99997, 99998, 99999]),
     array([0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0]),
     array([1, 1, 1, 1, 1, ..., 1, 1, 1, 1, 1]),
     array([3, 3, 3, 3, 3, ..., 3, 3, 3, 3, 3]))


    In [23]: a.f.record[:,:,1,3].view(np.int32)
    Out[23]: 
    array([[1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      17820, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           ...,
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      17337, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0],
           [1065353216,      39216,      39216,      39216,      39216, ...,          0,          0,          0,          0,          0]], dtype=int32)

    In [24]: a.f.record[:,0,1,3].view(np.int32)
    Out[24]: array([1065353216, 1065353216, 1065353216, 1065353216, 1065353216, ..., 1065353216, 1065353216, 1065353216, 1065353216, 1065353216], dtype=int32)

    In [25]: np.all( a.f.record[:,0,1,3].view(np.int32) == 1065353216 )
    Out[25]: True


Hmm, probably the 1. is coming from the inphoton::

    In [1]: a.f.inphoton[0]
    Out[1]: 
    array([[-12052.896,   9510.562,  11538.329,      0.1  ],
           [    -0.621,      0.49 ,      0.611,      1.   ],
           [    -0.619,     -0.785,      0.   ,    440.   ],
           [     0.   ,      0.   ,      0.   ,      0.   ]], dtype=float32)

Try changing ana/input_photons.py::

     23 class InputPhotons(object):
     24     """
     25     The "WEIGHT" has never been used as such. 
     26     The (1,3) sphoton.h slot is used for the iindex integer, 
     27     hence set the "WEIGHT" to 0.f in order that the int32 
     28     becomes zero. 
     29 
     30     np.zeros([1], dtype=np.float32 ).view(np.int32)[0] == 0 
     31     np.ones([1], dtype=np.float32 ).view(np.int32)[0] == 1065353216 
     32 
     33     See ~/opticks/notes/issues/sphoton_float_one_in_int32_1_3_iindex_slot.rst
     34     """
     35 
     36     WEIGHT = 0.  #  np.zeros([1], dtype=np.float32 ).view(np.int32)[0] == 0  
     37     
     38     DEFAULT_BASE = os.path.expanduser("~/.opticks/InputPhotons")
     39     DTYPE = np.float64 if os.environ.get("DTYPE","np.float32") == "np.float64" else np.float32
     40     
     41     X = np.array( [1., 0., 0.], dtype=DTYPE )
     42     Y = np.array( [0., 1., 0.], dtype=DTYPE )
     43     Z = np.array( [0., 0., 1.], dtype=DTYPE )


Will need to recrease all the InputPhotons for that to take effect::

    ls $HOME/.opticks/InputPhotons/Rain*



