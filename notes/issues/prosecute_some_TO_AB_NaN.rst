prosecute_some_TO_AB_NaN
===========================


provenance
-------------

* :doc:`tboolean_box_perfect_alignment`

::


    tboolean-;TBOOLEAN_TAG=3 tboolean-box --okg4 --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero



FIXED : by skipping the flags in the dv comparison
-----------------------------------------------------

ana/dv.py::

    149     def dv_(self, i, sel, lcu):
    150         ab = self.ab
    151         if self.name == "rpost_dv":
    152             av = ab.a.rpost()
    153             bv = ab.b.rpost()
    154         elif self.name == "rpol_dv":
    155             av = ab.a.rpol()
    156             bv = ab.b.rpol()
    157         elif self.name == "ox_dv":
    158             av = ab.a.ox[:,:3,:]
    159             bv = ab.b.ox[:,:3,:]
    160         else:
    161             assert self.name


The NaN are actually some int viewed as a float : so just need to skip the flags from the comparison

::

    In [26]: ab.a.ox
    Out[26]: 
    A()sliced
    A([[[  32.3058,  -30.8331, -380.7584,    0.4306],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]],

           [[ -14.9707,   25.2698, -282.4095,    0.7587],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]],

           [[ -32.0471,    6.9484, -223.9958,    0.9535],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]]], dtype=float32)

    In [27]: ab.a.ox[:,:3,:]
    Out[27]: 
    A()sliced
    A([[[  32.3058,  -30.8331, -380.7584,    0.4306],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ]],

           [[ -14.9707,   25.2698, -282.4095,    0.7587],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ]],

           [[ -32.0471,    6.9484, -223.9958,    0.9535],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ]]], dtype=float32)




rerun the 3 photons with NaNs
-------------------------------

::


   tboolean-;TBOOLEAN_TAG=3 tboolean-box-ip

    In [17]: ab.a.sel = "TO AB"

    In [18]: ab.a.where
    Out[18]: array([37922, 61642, 92906])

    In [19]: ab.b.sel = "TO AB"

    In [20]: ab.b.where
    Out[20]: array([37922, 61642, 92906])



::

    tboolean-;TBOOLEAN_TAG=4 tboolean-box --okg4 --align --mask 37922 --pindex 0 --pindexlog -DD --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero


::

    delta:optixrap blyth$ cat /tmp/blyth/opticks/ox_37922.log
    generate photon_id 0 
    WITH_ALIGN_DEV_DEBUG photon_id:0 bounce:0 
    propagate_to_boundary  u_OpBoundary:0.837183177 speed:299.79245 
    propagate_to_boundary  u_OpRayleigh:0.474008411   scattering_length(s.material1.z):1000000 scattering_distance:746530.188 
    propagate_to_boundary  u_OpAbsorption:0.999993086   absorption_length(s.material1.y):10000000 absorption_distance:69.1416245 
     WITH_ALIGN_DEV_DEBUG psave (32.3058167 -30.8330688 -380.758362 0.430631638) ( -2, 0, 67305985, 4104 ) 
    delta:optixrap blyth$ 
    delta:optixrap blyth$ 


Hmm so the NaN is actually int -2 viewed as a float : so just need to skip the flags from the comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    In [21]: np.int32(-2).view(np.float32)
    Out[21]: nan

    In [22]: np.int32(-1).view(np.float32)
    Out[22]: nan

    In [23]: np.int32(0).view(np.float32)
    Out[23]: 0.0

    In [24]: np.int32(1).view(np.float32)
    Out[24]: 1.4012985e-45

    In [25]: np.int32(2).view(np.float32)
    Out[25]: 2.8025969e-45





Find the NaN
---------------
   
Hmm photons that get absorbed prior to hitting anything have NaN in p.flags.f.x 

::

    In [3]: ab.ox_dv.dvs[14]
    Out[3]:  0014            :                          TO AB :       3        3  :         3      48/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    

    In [4]: ab.ox_dv.dvs[14].__class__
    Out[4]: opticks.ana.dv.Dv

    In [5]: dv = ab.ox_dv.dvs[14]

    In [6]: dv.av    
    Out[6]: 
    A()sliced
    A([[[  32.3058,  -30.8331, -380.7584,    0.4306],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]],

           [[ -14.9707,   25.2698, -282.4095,    0.7587],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]],

           [[ -32.0471,    6.9484, -223.9958,    0.9535],
            [  -0.    ,   -0.    ,    1.    ,    1.    ],
            [   0.    ,   -1.    ,    0.    ,  380.    ],
            [      nan,    0.    ,    0.    ,    0.    ]]], dtype=float32)

::

     71 __device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset)
     72 {
     73     pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );
     74     pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
     75     pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
     76     pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w);
     77 }

::

    156 #define FLAGS(p, s, prd) \
    157 { \
    158     p.flags.i.x = prd.boundary ;  \
    159     p.flags.u.y = s.identity.w ;  \
    160     p.flags.u.w |= s.flag ; \
    161 } \
    162 




