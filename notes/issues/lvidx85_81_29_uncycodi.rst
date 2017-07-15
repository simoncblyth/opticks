lvidx 85 81 29 uncycodi
=========================


::

     NScanTest

     nzero    3 count    4 frac 0.0160643
     i  163 nzero    3 NScanTest   85 soname                    OcrCalLsoPrt0xc1076b0 tag    [ 0:di] nprim    3 typ union difference cylinder cone  msg 
     i  167 nzero    3 NScanTest   81 soname                    OcrGdsLsoPrt0xc104978 tag    [ 0:di] nprim    3 typ union difference cylinder cone  msg 
     i  212 nzero    3 NScanTest   36 soname                       IavTopRib0xbf8e168 tag    [ 0:di] nprim    3 typ difference cone box3  msg 
     i  219 nzero    3 NScanTest   29 soname                       OcrGdsPrt0xc352518 tag    [ 0:di] nprim    3 typ union difference cylinder cone  msg 

     ## 4 real problem solids
     ## 3 are same issue : (cy+cy)-co
        
         opticks-tbool-vi 85  : union of cylinders with cone subtracted (base of cone coincident with base of one cylinder)
         opticks-tbool-vi 81  : ditto 
         opticks-tbool-vi 29  : ditto 

     ##  looks to be from bx-bx due to same y-dim
         opticks-tbool-vi 36  :   thin artifact edge changes depending on view : (bx-bx)-co      y-dim of subtracted boxes are same
        


MAXMIN between union siblings
--------------------------------

MAXMIN can be treated by znudge lineup if siblings of UNION.

Handling MINMIN with difference...
------------------------------------

For MINMIN the answer is known in this case : expand the cone down 

* but its unclear how to know in general that it is a safe change (does not change geometry),

* would a composite SDF > 0 (outside) in the region be sufficient to know that 
  are not bumping into anything by expand a subtracted prim 

* pick the prim with difference parent and assume that its being subtracted
  (actually should +ize as standard first and then check complement
  to robustly know that the territory is not in the composite... but not ready for that)
  
  * could check SDF above and below the coincidence ... should both be  +ve 



::

    OcrCalLsoPrt0xc1076b0 union difference cylinder cone  verbosity 3 root.treeidx  85 num_prim  3 num_coincidence  2 MINMIN  1 MINMAX  0 MAXMIN  1 MAXMAX  0
    ( 3, 4) PAIR_MAXMIN [ 3:cy] [ 4:cy]
    ( 3, 2) PAIR_MINMIN [ 3:cy] [ 2:co]

    OcrGdsLsoPrt0xc104978 union difference cylinder cone  verbosity 3 root.treeidx  81 num_prim  3 num_coincidence  2 MINMIN  1 MINMAX  0 MAXMIN  1 MAXMAX  0
    ( 3, 4) PAIR_MAXMIN [ 3:cy] [ 4:cy]
    ( 3, 2) PAIR_MINMIN [ 3:cy] [ 2:co]




