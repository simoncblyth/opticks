photon-polarization-testauto-SR
==================================


ISSUE : testauto giving NaN polarizaton for SR
-------------------------------------------------

Getting NaN in photon polarization::






APPROACH
----------

Narrow autoemitconfig uv domain such that all photons will SR
and SC AB are switched off

* note that the autoemitconfig option must be given to the python geometry prep stage, 
  not the OKG4Test executable

::

     tboolean-;tboolean-box --okg4 --testauto --noab --nosc 


::

     710 tboolean-box--(){ cat << EOP 
     711 import logging
     712 log = logging.getLogger(__name__)
     713 from opticks.ana.base import opticks_main
     714 from opticks.analytic.polyconfig import PolyConfig
     715 from opticks.analytic.csg import CSG  
     716 
     717 autoemitconfig="photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.45,umax:0.55,vmin:0.45,vmax:0.55"
     718 args = opticks_main(csgpath="$TMP/$FUNCNAME", autoemitconfig=autoemitconfig)
     719 
     720 emitconfig = "photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75" 
     721 
     722 CSG.kwa = dict(poly="IM",resolution="20", verbosity="0",ctrl="0", containerscale="3", emitconfig=emitconfig  )
     723 
     724 container = CSG("box", emit=-1, boundary='Rock//perfectAbsorbSurface/Vacuum', container="1" )  # no param, container="1" switches on auto-sizing
     725 
     726 box = CSG("box3", param=[300,300,200,0], emit=0,  boundary="Vacuum///GlassSchottF2" )
     727 
     728 CSG.Serialize([container, box], args )
     729 EOP
     730 }


cu/propagate.h DEBUG_POLZ::

    2017-12-01 13:22:15.641 INFO  [832957] [OPropagator::prelaunch@166] 1 : (0;10,1) prelaunch_times vali,comp,prel,lnch  0.0001 3.4463 0.1303 0.0000
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.0 polz (    0.0000    -1.0000     0.0000) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    // propagate_at_specular_reflector.1 polz (       nan        nan        nan) 
    2017-12-01 13:22:15.655 INFO  [832957] [OContext::launch@322] OContext::launch LAUNCH time: 0.01389




::

    2017-12-01 13:05:45,200] p54370 {/Users/blyth/opticks/ana/ab.py:156} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1305 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1305 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              8ad    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO SR SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000             1280    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] TO|SR|SA
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    0000              122    600000    600000             0.00        1.000 +- 0.001        1.000 +- 0.001  [3 ] Vm Vm Rk
    .                             600000    600000         0.00/0 =  0.00  (pval:nan prob:nan)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 edfd1a210c3da6e4b725d3e4c2a2a59e 88d3ee8cc1674e4766a5b293d552ca26  600000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.013763847773677895] 
     0000            :                       TO SR SA :  600000   600000  :    600000 7200000/     18: 0.000  mx/mn/av 0.01376/     0/3.441e-08  eps:0.0002    
    rpol_dv maxdvmax:2.0 maxdv:[2.0] 
     0000            :                       TO SR SA :  600000   600000  :    600000 5400000/3000000: 0.556  mx/mn/av      2/     0/0.6667  eps:0.0002    
    /Users/blyth/opticks/ana/dv.py:58: RuntimeWarning: invalid value encountered in greater
      discrep = dv[dv>eps]
    ox_dv maxdvmax:nan maxdv:[nan] 
     0000            :                       TO SR SA :  600000   600000  :    600000 9600000/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    
    c2p : {'seqmat_ana': 0.0, 'pflags_ana': 0.0, 'seqhis_ana': 0.0} c2pmax: 0.0  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 2.0, 'rpost_dv': 0.013763847773677895} rmxs_max_: 2.0  CUT ok.rdvmax 0.1  RC:88 
    pmxs_ : {'ox_dv': nan} pmxs_max_: nan  CUT ok.pdvmax 0.001  RC:88 





::

    [2017-12-01 12:35:15,285] p50967 {/Users/blyth/opticks/ana/ab.py:156} INFO - AB.init_point DONE
    AB(1,torch,tboolean-box)  None 0 
    A tboolean-box/torch/  1 :  20171201-1233 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/1/fdom.npy () 
    B tboolean-box/torch/ -1 :  20171201-1233 maxbounce:9 maxrec:10 maxrng:3000000 /tmp/blyth/opticks/evt/tboolean-box/torch/-1/fdom.npy (recstp) 
    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    /tmp/blyth/opticks/tboolean-box--
    .                seqhis_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         1.12/5 =  0.22  (pval:0.953 prob:0.047)  
    0000               8d    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO SA
    0001              8ad    207533    207524             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO SR SA
    0002              86d       368       368             0.00        1.000 +- 0.052        1.000 +- 0.052  [3 ] TO SC SA
    0003             8a6d        58        64             0.30        0.906 +- 0.119        1.103 +- 0.138  [4 ] TO SC SR SA
    0004             86ad        50        42             0.70        1.190 +- 0.168        0.840 +- 0.130  [4 ] TO SR SC SA
    0005               4d        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] TO AB
    0006            8a6ad         6        10             0.00        0.600 +- 0.245        1.667 +- 0.527  [5 ] TO SR SC SR SA
    0007              4ad         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] TO SR AB
    .                             600000    600000         1.12/5 =  0.22  (pval:0.953 prob:0.047)  
    .                pflags_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.14/4 =  0.04  (pval:0.998 prob:0.002)  
    0000             1080    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] TO|SA
    0001             1280    207533    207524             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] TO|SR|SA
    0002             10a0       368       368             0.00        1.000 +- 0.052        1.000 +- 0.052  [3 ] TO|SA|SC
    0003             12a0       114       116             0.02        0.983 +- 0.092        1.018 +- 0.094  [4 ] TO|SR|SA|SC
    0004             1008        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] TO|AB
    0005             1208         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] TO|SR|AB
    .                             600000    600000         0.14/4 =  0.04  (pval:0.998 prob:0.002)  
    .                seqmat_ana  1:tboolean-box   -1:tboolean-box        c2        ab        ba 
    .                             600000    600000         0.15/3 =  0.05  (pval:0.986 prob:0.014)  
    0000               12    391943    391952             0.00        1.000 +- 0.002        1.000 +- 0.002  [2 ] Vm Rk
    0001              122    207901    207892             0.00        1.000 +- 0.002        1.000 +- 0.002  [3 ] Vm Vm Rk
    0002             1222       108       106             0.02        1.019 +- 0.098        0.981 +- 0.095  [4 ] Vm Vm Vm Rk
    0003               22        37        34             0.13        1.088 +- 0.179        0.919 +- 0.158  [2 ] Vm Vm
    0004            12222         6        10             0.00        0.600 +- 0.245        1.667 +- 0.527  [5 ] Vm Vm Vm Vm Rk
    0005              222         5         6             0.00        0.833 +- 0.373        1.200 +- 0.490  [3 ] Vm Vm Vm
    .                             600000    600000         0.15/3 =  0.05  (pval:0.986 prob:0.014)  



ISSUE : propagate_at_specular_reflector giving NaN polz
----------------------------------------------------------


cu/generate.cu::

    516 
    517         command = propagate_to_boundary( p, s, rng );
    518         if(command == BREAK)    break ;           // BULK_ABSORB
    519         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    520         // PASS : survivors will go on to pick up one of the below flags, 
    521 
    522         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    523         {
    524             command = propagate_at_surface(p, s, rng);
    525             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    526             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    527         }
    528         else
    529         {
    530             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    531             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    532             // tacit CONTINUE
    533         }



cu/propagate.h::

    518 __device__ int
    519 propagate_at_surface(Photon &p, State &s, curandState &rng)
    520 {
    521 
    522     float u = curand_uniform(&rng);
    523 
    524     if( u < s.surface.y )   // absorb   
    525     {
    526         s.flag = SURFACE_ABSORB ;
    527         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    528         return BREAK ;
    529     }
    530     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    531     {
    532         s.flag = SURFACE_DETECT ;
    533         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    534         return BREAK ;
    535     }
    536     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    537     {
    538         s.flag = SURFACE_DREFLECT ;
    539         propagate_at_diffuse_reflector_geant4_style(p, s, rng);
    540         return CONTINUE;
    541     }
    542     else
    543     {
    544         s.flag = SURFACE_SREFLECT ;
    545         propagate_at_specular_reflector(p, s, rng );
    546         return CONTINUE;
    547     }
    548 }
    549 



::

    413 __device__ void propagate_at_specular_reflector(Photon &p, State &s, curandState &rng)
    414 {
    415     const float c1 = -dot(p.direction, s.surface_normal );     // c1 arranged to be +ve   
    416 
    417     // TODO: make change to c1 for normal incidence detection
    418 
    419     float3 incident_plane_normal = fabs(s.cos_theta) < 1e-6f ? p.polarization : normalize(cross(p.direction, s.surface_normal)) ;
    420 
    421     float normal_coefficient = dot(p.polarization, incident_plane_normal);  // fraction of E vector perpendicular to plane of incidence, ie S polarization
    422 
    423     p.direction += 2.0f*c1*s.surface_normal  ;
    424 
    425     bool s_polarized = curand_uniform(&rng) < normal_coefficient*normal_coefficient ;
    426 
    427     p.polarization = s_polarized
    428                        ?
    429                           incident_plane_normal
    430                        :
    431                           normalize(cross(incident_plane_normal, p.direction))
    432                        ;
    433 
    434     p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    435 }





All final photon polz in "TO SR SA" are NaN
---------------------------------------------

::

    simon:opticks blyth$ tboolean-;tboolean-box-ip

    In [2]: ab.aselhis = "TO SR SA"

    In [3]: ab.a.ox
    Out[3]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

    In [6]: ab.a.ox[:,2,:3]
    Out[6]: 
    A()sliced
    A([[ nan,  nan,  nan],
           [ nan,  nan,  nan],
           [ nan,  nan,  nan],
           ..., 
           [ nan,  nan,  nan],
           [ nan,  nan,  nan],
           [ nan,  nan,  nan]], dtype=float32)

    In [7]: np.isnan(ab.a.ox[:,2,:3])
    Out[7]: 
    A()sliced
    A([[ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True],
           ..., 
           [ True,  True,  True],
           [ True,  True,  True],
           [ True,  True,  True]], dtype=bool)

    In [8]: np.all(np.isnan(ab.a.ox[:,2,:3]))
    Out[8]: 
    A()sliced
    A(True, dtype=bool)




Point-by-point pol are unset beyond first point::

    In [4]: ab.a.rpol()
    Out[4]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],






Confirmed that NaN polz issue is specific to testauto/SR
------------------------------------------------------------

::

    simon:opticks blyth$ tboolean-;tboolean-box --okg4 
    ...

    .                             100000    100000         1.61/4 =  0.40  (pval:0.807 prob:0.193)  
    ab.a.metadata                  /tmp/blyth/opticks/evt/tboolean-box/torch/1 8210ebdae5967a9ef905291542364a4b 54be6772c3093360d09fefc4346e74a0  100000    -1.0000 INTEROP_MODE 
    ab.a.metadata.csgmeta0 {u'containerscale': u'3', u'container': u'1', u'ctrl': u'0', u'verbosity': u'0', u'poly': u'IM', u'emitconfig': u'photons:100000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x1,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75', u'resolution': u'20', u'emit': -1}
    rpost_dv maxdvmax:0.0137638477737 maxdv:[0.0, 0.013763847773674343, 0.0, 0.0, 0.0] 
     0000            :                          TO SA :   55321    55303  :     55249  441992/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  551872/      8: 0.000  mx/mn/av 0.01376/     0/1.995e-07  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    2256/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    2500/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      24/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    rpol_dv maxdvmax:0.0 maxdv:[0.0, 0.0, 0.0, 0.0, 0.0] 
     0000            :                          TO SA :   55321    55303  :     55249  331494/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  413904/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    1692/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    1875/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      18/      0: 0.000  mx/mn/av      0/     0/     0  eps:0.0002    
    ox_dv maxdvmax:3.0517578125e-05 maxdv:[3.0517578125e-05, 5.960464477539063e-08, 1.401298464324817e-45, 5.960464477539063e-08, 5.960464477539063e-08] 
     0000            :                          TO SA :   55321    55303  :     55249  883984/      0: 0.000  mx/mn/av 3.052e-05/     0/1.907e-06  eps:0.0002    
     0001            :                    TO BT BT SA :   39222    39231  :     34492  551872/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0002            :                       TO BR SA :    2768     2814  :       188    3008/      0: 0.000  mx/mn/av 1.401e-45/     0/8.758e-47  eps:0.0002    
     0003            :                 TO BT BR BT SA :    2425     2369  :       125    2000/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
     0004            :              TO BT BR BR BT SA :     151      142  :         1      16/      0: 0.000  mx/mn/av 5.96e-08/     0/3.725e-09  eps:0.0002    
    c2p : {'seqmat_ana': 0.40311601124980434, 'pflags_ana': 1.0829369776001112, 'seqhis_ana': 0.88772768790641765} c2pmax: 1.0829369776  CUT ok.c2max 2.0  RC:0 
    rmxs_ : {'rpol_dv': 0.0, 'rpost_dv': 0.013763847773674343} rmxs_max_: 0.0137638477737  CUT ok.rdvmax 0.1  RC:0 
    pmxs_ : {'ox_dv': 3.0517578125e-05} pmxs_max_: 3.0517578125e-05  CUT ok.pdvmax 0.001  RC:0 
    [2017-12-01 12:27:18,399] p49848 {/Users/blyth/opticks/ana/tboolean.py:43} INFO - early exit as non-interactive




Saving into photon buffer
--------------------------


     71 __device__ void psave( Photon& p, optix::buffer<float4>& pbuffer, unsigned int photon_offset)
     72 {
     73     pbuffer[photon_offset+0] = make_float4( p.position.x,    p.position.y,    p.position.z,     p.time );
     74     pbuffer[photon_offset+1] = make_float4( p.direction.x,   p.direction.y,   p.direction.z,    p.weight );
     75     pbuffer[photon_offset+2] = make_float4( p.polarization.x,p.polarization.y,p.polarization.z, p.wavelength );
     76     pbuffer[photon_offset+3] = make_float4( p.flags.f.x,     p.flags.f.y,     p.flags.f.z,      p.flags.f.w);
     77 }
     78 



::

    tboolean-;tboolean-box --okg4 --testauto
    tboolean-;tboolean-box-ip

    In [2]: ab.dvtabs[2]
    Out[2]: 
    ox_dv maxdvmax:3.0517578125e-05 maxdv:[3.0517578125e-05, nan] 
     0000            :                          TO SA :  391943   391952  :    391558 6264928/      0: 0.000  mx/mn/av 3.052e-05/     0/1.907e-06  eps:0.0002    
     0001            :                       TO SR SA :  207533   207524  :    207394 3318304/      0: 0.000  mx/mn/av    nan/   nan/   nan  eps:0.0002    


    In [8]: dvt.dvs[1].av
    Out[8]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -44.3963, -116.7347, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -43.5826, -147.5403, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           ..., 
           [[-144.0839,  450.    ,  -23.8085,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[  71.1732,  450.    ,   56.2633,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -91.8347,  450.    ,   29.8083,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [      nan,       nan,       nan,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]]], dtype=float32)

    In [9]: dvt.dvs[1].bv
    Out[9]: 
    A()sliced
    A([[[-133.4443,   -1.4124, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -44.3963, -116.7347, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -43.5826, -147.5403, -450.    ,    2.5346],
            [   0.    ,    0.    ,   -1.    ,    1.    ],
            [   0.    ,    1.    ,    0.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           ..., 
           [[-144.0839,  450.    ,  -23.8085,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[  71.1732,  450.    ,   56.2633,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]],

           [[ -91.8347,  450.    ,   29.8083,    2.2011],
            [   0.    ,    1.    ,    0.    ,    1.    ],
            [   0.    ,    0.    ,    1.    ,  380.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ]]], dtype=float32)

    In [10]: 



::


    In [16]: ab.a.ox[:20,2]
    Out[16]: 
    A()sliced
    A([[   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [   0.,   -1.,    0.,  380.],
           [   0.,   -1.,    0.,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.]], dtype=float32)

    In [18]: ab.a.ox.shape
    Out[18]: (600000, 4, 4)

    In [20]: ab.a.seqhis.shape
    Out[20]: (600000,)

    In [21]: ab.a.seqhis[:20]
    Out[21]: 
    A()sliced
    A([ 141, 2221,  141,  141,  141,  141, 2221, 2221,  141, 2221,  141,  141,  141,  141,  141, 2221,  141,  141, 2221, 2221], dtype=uint64)

    In [22]: hex(2221)
    Out[22]: '0x8ad'


    In [23]: ab.selhis = "TO SR SA"

    In [25]: ab.a.ox[:20,2]
    Out[25]: 
    A()sliced
    A([[  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.],
           [  nan,   nan,   nan,  380.]], dtype=float32)

    In [27]: ab.a.ox.shape
    Out[27]: (207533, 4, 4)

    In [28]: ab.a.rpol()
    Out[28]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0., -1.,  0.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           ..., 
           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]],

           [[ 0.,  0., -1.],
            [-1., -1., -1.],
            [-1., -1., -1.]]], dtype=float32)

    In [29]: ab.b.rpol()
    Out[29]: 
    A()sliced
    A([[[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           [[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           [[ 0., -1.,  0.],
            [ 0.,  1.,  0.],
            [ 0.,  1.,  0.]],

           ..., 
           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]],

           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]],

           [[ 0.,  0., -1.],
            [ 0.,  0.,  1.],
            [ 0.,  0.,  1.]]], dtype=float32)



