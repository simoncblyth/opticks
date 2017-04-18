NCSG Translated PMT
======================


Cathode Inner or Outer
---------------------------

* can see from front but disappearing from back 
* observe wierdness in t_min clipping, 

* testing with tboolean-zsphere see the same wierdness, 
  its the missing cap handling 

* intersecting with a zslab works, but then you get a cap 

* used a flag to switch off the cap, but now getting sliver artifact and 
  spurious intersects

* actually switching off the caps prevents slab intersection from working, 
  get nothing with tboolean-sphere-slab ... cannot selectively have the intersect work for doing 
  the intersection chop and not work for giving an open cap...

  * cannot use infinite slab intersection without enabling the caps

  * so cannot use slab intersection and have open caps 
  * hmm, means must implement cap handling similar to cylinder in zsphere


Testing with tboolean-pmt with a kludge to just 
return the inner or outer in ncsgtranslator.py::


    182         cn.param[0] = en.xyz[0]
    183         cn.param[1] = en.xyz[1]
    184         cn.param[2] = en.xyz[2]
    185         cn.param[3] = radius
    186 
    187         if has_inner:
    188             #ret = CSG("difference", left=cn, right=inner )
    189             ret = inner
    190         else:
    191             ret = cn
    192         pass
    193         return ret
    194 






::

    2017-04-18 18:43:57.920 INFO  [962828] [GParts::dump@857] GParts::dump ni 4
         0.0000      0.0000      0.0000   1000.0000 
         0.0000      0.0000     123 <-bnd        0 <-INDEX    bn Rock//perfectAbsorbSurface/Vacuum 
         0.0000      0.0000      0.0000           6 (box) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000     124 <-bnd        1 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000      0.0000    127.9500 
        97.2867    127.9500     124 <-bnd        2 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000     43.0000     98.9500 
        12.9934     55.7343     124 <-bnd        3 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

