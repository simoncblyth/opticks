SURFACE_ABSORB
=================


NEXT
------

* logical as opposed to border surfaces need to be repeated in ISUR and OSUR slots for Opticks
  to match the G4 logic, although expect no impact of this for almost all surfaces




dbgseqhis
----------

::

   tlaser-t --dbgseqhis 8cccc6d



Concentric test
------------------


Compare with tdefault where photons distributed randomly from point source
-----------------------------------------------------------------------------

Note that "8cccc6d" is matching better when average over all directions
as compared to tlaser which picks one. This supports the cause being triangulation.

::

    /Users/blyth/opticks/ana/tdefault.py
    [2016-10-31 15:13:16,021] p6238 {/Users/blyth/opticks/ana/tdefault.py:24} INFO - tag 1 src torch det default c2max 2.0  
    [2016-10-31 15:13:16,588] p6238 {/Users/blyth/opticks/ana/cf.py:36} INFO - CF a default/torch/  1 :  20161031-1330 /tmp/blyth/opticks/evt/default/torch/1/fdom.npy 
    [2016-10-31 15:13:16,588] p6238 {/Users/blyth/opticks/ana/cf.py:37} INFO - CF b default/torch/ -1 :  20161031-1330 /tmp/blyth/opticks/evt/default/torch/-1/fdom.npy 
              seqhis_ana   1:default   -1:default           c2           ab           ba 
                  8ccccd         46457        46236             0.53         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
              ccaccccccd          9547         8240            96.04         1.16 +- 0.01         0.86 +- 0.01  [10] TO BT BT BT BT BT BT SR BT BT
              cccacccccd          1194         6627          3774.13         0.18 +- 0.01         5.55 +- 0.07  [10] TO BT BT BT BT BT SR BT BT BT
                 4cccccd          6611          527          5185.63        12.54 +- 0.15         0.08 +- 0.00  [7 ] TO BT BT BT BT BT AB
                      4d          5008         5021             0.02         1.00 +- 0.01         1.00 +- 0.01  [2 ] TO AB
                 8cccccd          3231         4623           246.71         0.70 +- 0.01         1.43 +- 0.02  [7 ] TO BT BT BT BT BT SA
              cccbcccccd          4098         4548            23.42         0.90 +- 0.01         1.11 +- 0.02  [10] TO BT BT BT BT BT BR BT BT BT
               8cbcccccd          2728         2394            21.78         1.14 +- 0.02         0.88 +- 0.02  [9 ] TO BT BT BT BT BT BR BT SA

                 8cccc6d          2084         1939             5.23         1.07 +- 0.02         0.93 +- 0.02  [7 ] TO SC BT BT BT BT SA

                8ccccccd          1677         1107           116.70         1.51 +- 0.04         0.66 +- 0.02  [8 ] TO BT BT BT BT BT BT SA
              cacccccccd           279         1591           920.50         0.18 +- 0.01         5.70 +- 0.14  [10] TO BT BT BT BT BT BT BT SR BT
              cccc9ccccd          1455         1431             0.20         1.02 +- 0.03         0.98 +- 0.03  [10] TO BT BT BT BT DR BT BT BT BT
                    4ccd          1168         1277             4.86         0.91 +- 0.03         1.09 +- 0.03  [4 ] TO BT BT AB
                 7cccccd           657         1116           118.83         0.59 +- 0.02         1.70 +- 0.05  [7 ] TO BT BT BT BT BT SD
              accccccccd           896           19           840.58        47.16 +- 1.58         0.02 +- 0.00  [10] TO BT BT BT BT BT BT BT BT SR
                 8cccc5d           863          850             0.10         1.02 +- 0.03         0.98 +- 0.03  [7 ] TO RE BT BT BT BT SA
              abaccccccd           314          797           209.98         0.39 +- 0.02         2.54 +- 0.09  [10] TO BT BT BT BT BT BT SR BR SR
                  4ccccd           736          773             0.91         0.95 +- 0.04         1.05 +- 0.04  [6 ] TO BT BT BT BT AB
               4cccccccd           586           10           556.67        58.60 +- 2.42         0.02 +- 0.01  [9 ] TO BT BT BT BT BT BT BT AB
              ccccbccccd           136          481           192.91         0.28 +- 0.02         3.54 +- 0.16  [10] TO BT BT BT BT BR BT BT BT BT
                              100000       100000       101.27 


Complement checks... 
----------------------

Using new ana dbgseqhis option, can see the totals for the complement to a sequence.
The Opticks shortfall in SA is made up with more BT.
From this it seems plausible that the difference could be the result of 
triangulated vs analytic geometry, due to different angle at which the scatter 
from the laser beam impinges on triangulated vs analytic surfaces. 

Easiest way to verify is to construct a concentric sphere
test geometry, bounded by RSOilSurface.

So aiming for a test geometry that has all the features of the real one
other than the complicated geometry.


::

    simon:ana blyth$ tlaser.py --dbgseqhis cccc6d
    /Users/blyth/opticks/ana/tlaser.py --dbgseqhis cccc6d
    [2016-10-31 12:46:11,821] p5110 {/Users/blyth/opticks/ana/tlaser.py:25} INFO - tag 1 src torch det laser c2max 2.0  
    [2016-10-31 12:46:12,272] p5110 {/Users/blyth/opticks/ana/tlaser.py:48} INFO -  a : laser/torch/  1 :  20161031-1151 /tmp/blyth/opticks/evt/laser/torch/1/fdom.npy 
    [2016-10-31 12:46:12,272] p5110 {/Users/blyth/opticks/ana/tlaser.py:49} INFO -  b : laser/torch/ -1 :  20161031-1151 /tmp/blyth/opticks/evt/laser/torch/-1/fdom.npy 
              seqhis_ana     1:laser     -1:laser           c2           ab           ba 
                 8cccc6d          1570         1846            22.30         0.85 +- 0.02         1.18 +- 0.03  [7 ] TO SC BT BT BT BT SA
              cacccccc6d           312          247             7.56         1.26 +- 0.07         0.79 +- 0.05  [10] TO SC BT BT BT BT BT BT SR BT
                4ccccc6d           257            4           245.25        64.25 +- 4.01         0.02 +- 0.01  [8 ] TO SC BT BT BT BT BT AB
                8ccccc6d           100          223            46.84         0.45 +- 0.04         2.23 +- 0.15  [8 ] TO SC BT BT BT BT BT SA
              ccbccccc6d           148          124             2.12         1.19 +- 0.10         0.84 +- 0.08  [10] TO SC BT BT BT BT BT BR BT BT
              ccaccccc6d            24          131            73.86         0.18 +- 0.04         5.46 +- 0.48  [10] TO SC BT BT BT BT BT SR BT BT
              8cbccccc6d            84           80             0.10         1.05 +- 0.11         0.95 +- 0.11  [10] TO SC BT BT BT BT BT BR BT SA
              cccccccc6d            73           32            16.01         2.28 +- 0.27         0.44 +- 0.08  [10] TO SC BT BT BT BT BT BT BT BT
               8cccccc6d            66           33            11.00         2.00 +- 0.25         0.50 +- 0.09  [9 ] TO SC BT BT BT BT BT BT SA
              ccc9cccc6d            50           55             0.24         0.91 +- 0.13         1.10 +- 0.15  [10] TO SC BT BT BT BT DR BT BT BT
                7ccccc6d            19           55            17.51         0.35 +- 0.08         2.89 +- 0.39  [8 ] TO SC BT BT BT BT BT SD
              accccccc6d            52           34             3.77         1.53 +- 0.21         0.65 +- 0.11  [10] TO SC BT BT BT BT BT BT BT SR
              cccbcccc6d            51           15            19.64         3.40 +- 0.48         0.29 +- 0.08  [10] TO SC BT BT BT BT BR BT BT BT
               4cccccc6d            43            2            37.36        21.50 +- 3.28         0.05 +- 0.03  [9 ] TO SC BT BT BT BT BT BT AB
                 4cccc6d            35           27             1.03         1.30 +- 0.22         0.77 +- 0.15  [7 ] TO SC BT BT BT BT AB
              cbcccccc6d            29           31             0.07         0.94 +- 0.17         1.07 +- 0.19  [10] TO SC BT BT BT BT BT BT BR BT
              bccccccc6d            29           22             0.96         1.32 +- 0.24         0.76 +- 0.16  [10] TO SC BT BT BT BT BT BT BT BR
              4ccccccc6d            28            0             0.00         0.00 +- 0.00         0.00 +- 0.00  [10] TO SC BT BT BT BT BT BT BT AB
              bacccccc6d            24           25             0.02         0.96 +- 0.20         1.04 +- 0.21  [10] TO SC BT BT BT BT BT BT SR BR
              abaccccc6d            13            8             0.00         1.62 +- 0.45         0.62 +- 0.22  [10] TO SC BT BT BT BT BT SR BR SR
                89cccc6d            11            6             0.00         1.83 +- 0.55         0.55 +- 0.22  [8 ] TO SC BT BT BT BT DR SA
                8bcccc6d             6           10             0.00         0.60 +- 0.24         1.67 +- 0.53  [8 ] TO SC BT BT BT BT BR SA
                86cccc6d             9            7             0.00         1.29 +- 0.43         0.78 +- 0.29  [8 ] TO SC BT BT BT BT SC SA
              9cbccccc6d             3            8             0.00         0.38 +- 0.22         2.67 +- 0.94  [10] TO SC BT BT BT BT BT BR BT DR
              ccc6cccc6d             4            7             0.00         0.57 +- 0.29         1.75 +- 0.66  [10] TO SC BT BT BT BT SC BT BT BT






Visual inspection of photon termination points
-----------------------------------------------

* comparing distrib of photon termination points
  using photon flag interface between tlaser-v and tlaser-vg4 
  shows no large discrep 


Surface info comparison with CInterpolationTest and OInterpolationTest
------------------------------------------------------------------------


* no smoking guns 



Commenting out ESRAir reflectivity diddle doesnt fix the 20%
----------------------------------------------------------------------

::

     335               if (PropertyPointer)
     336               {
     337 
     338 #if ( G4VERSION_NUMBER > 1000 )
     339                  theReflectivity = PropertyPointer->Value(thePhotonMomentum);
     340 #else
     341                  theReflectivity = PropertyPointer->GetProperty(thePhotonMomentum);
     342 #endif
     343 
     344                  if(OpticalSurface->GetName().contains("ESRAir"))
     345                  {
     346                       G4double inciAngle = GetIncidentAngle();
     347                       //ESR in air
     348                       if(inciAngle*180./pi > 40)
     349                       {
     350                           theReflectivity = (theReflectivity - 0.993) + 0.973572 + 9.53233e-04*(inciAngle*180./pi) - 1.22184e-05*((inciAngle*180./pi))*((inciAngle*180./pi));
     351                       }



1M 2016 Oct 28 seqhis
------------------------

In general the progressive mask totals show good step-by-step agreement, 
discrepancies coming in only at last step (AB or SA).

::


    [2016-10-28 11:16:32,771] p43831 {/Users/blyth/opticks/ana/tlaser.py:48} INFO -  a : laser/torch/  1 :  20161028-1116 /tmp/blyth/opticks/evt/laser/torch/1/fdom.npy 
    [2016-10-28 11:16:32,772] p43831 {/Users/blyth/opticks/ana/tlaser.py:49} INFO -  b : laser/torch/ -1 :  20161028-1116 /tmp/blyth/opticks/evt/laser/torch/-1/fdom.npy 
              seqhis_ana     1:laser     -1:laser           c2           ab           ba 
                  8ccccd        813163       813761             0.22         1.00 +- 0.00         1.00 +- 0.00  [6 ] TO BT BT BT BT SA
                      4d         45622        45617             0.00         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO AB
              cccc9ccccd         27443        27012             3.41         1.02 +- 0.01         0.98 +- 0.01  [10] TO BT BT BT BT DR BT BT BT BT
                 8cccc6d         15516        18592           277.41         0.83 +- 0.01         1.20 +- 0.01  [7 ] TO SC BT BT BT BT SA               ## ~20% final SA
                    4ccd         10975        11210             2.49         0.98 +- 0.01         1.02 +- 0.01  [4 ] TO BT BT AB
                  4ccccd          9002         8820             1.86         1.02 +- 0.01         0.98 +- 0.01  [6 ] TO BT BT BT BT AB
                 8cccc5d          8433         8284             1.33         1.02 +- 0.01         0.98 +- 0.01  [7 ] TO RE BT BT BT BT SA
                 8cc6ccd          3370         3943            44.90         0.85 +- 0.01         1.17 +- 0.02  [7 ] TO BT BT SC BT BT SA               ## ~20% final SA
              cacccccc6d          3345         2435           143.27         1.37 +- 0.02         0.73 +- 0.01  [10] TO SC BT BT BT BT BT BT SR BT      ## trunc
              cccccc6ccd          2930         2396            53.54         1.22 +- 0.02         0.82 +- 0.02  [10] TO BT BT SC BT BT BT BT BT BT      ## trunc
                 86ccccd          2554         2707             4.45         0.94 +- 0.02         1.06 +- 0.02  [7 ] TO BT BT BT BT SC SA               ## ~20% final SA
                     45d          2436         2490             0.59         0.98 +- 0.02         1.02 +- 0.02  [3 ] TO RE AB
                4ccccc6d          2431           78          2206.70        31.17 +- 0.63         0.03 +- 0.00  [8 ] TO SC BT BT BT BT BT AB            ## drastic AB discrep 

                   tlaser-v    shows the discrepant AB to be associated with specific geometry in viscinity of bottom reflector
                   tlaser-vg4  cannot show the 78 as does not make it into the top chart

                8cccc55d          2180         2119             0.87         1.03 +- 0.02         0.97 +- 0.02  [8 ] TO RE RE BT BT BT BT SA
                 89ccccd          2011         2152             4.78         0.93 +- 0.02         1.07 +- 0.02  [7 ] TO BT BT BT BT DR SA               ## final SA
              cccc6ccccd          2068         1750            26.49         1.18 +- 0.03         0.85 +- 0.02  [10] TO BT BT BT BT SC BT BT BT BT      ## trunc 
                   4cccd          2065         1990             1.39         1.04 +- 0.02         0.96 +- 0.02  [5 ] TO BT BT BT AB
                8ccccc6d           991         1985           332.00         0.50 +- 0.02         2.00 +- 0.04  [8 ] TO SC BT BT BT BT BT SA            ## final SA (OK is half of G4)
                 8cc5ccd          1898         1964             1.13         0.97 +- 0.02         1.03 +- 0.02  [7 ] TO BT BT RE BT BT SA
              ccbccccc6d          1621         1309            33.22         1.24 +- 0.03         0.81 +- 0.02  [10] TO SC BT BT BT BT BT BR BT BT      ## trunc
                             1000000      1000000        37.28 


     Progressive mask development of the 20% discrepant 8cccc6d  shows problem to be 
     all in final SURFACE_ABSORB SA step, with G4 absorbing 20% more than OK.
     Note that top line SA is in agreement, but 2nd step SC means are going in a 
     random direction, indicating an issue with the "average" absorbing surface 
     that is not present with the direct surface pointed at by the laser.

     tlaser-v shows no focus on any specific geometry.


                      6d         36156        35863             1.19         1.01 +- 0.01         0.99 +- 0.01  [2 ] TO SC
                     c6d         32422        32101             1.60         1.01 +- 0.01         0.99 +- 0.01  [3 ] TO SC BT
                    cc6d         32333        32014             1.58         1.01 +- 0.01         0.99 +- 0.01  [4 ] TO SC BT BT
                   ccc6d         31049        30857             0.60         1.01 +- 0.01         0.99 +- 0.01  [5 ] TO SC BT BT BT
                  cccc6d         30884        30721             0.43         1.01 +- 0.01         0.99 +- 0.01  [6 ] TO SC BT BT BT BT
                 8cccc6d         15516        18592           277.41         0.83 +- 0.01         1.20 +- 0.01  [7 ] TO SC BT BT BT BT SA

     Same again issue with final SA.

                      cd        892640       893243             0.20         1.00 +- 0.00         1.00 +- 0.00  [2 ] TO BT
                     ccd        891267       891910             0.23         1.00 +- 0.00         1.00 +- 0.00  [3 ] TO BT BT
                    6ccd          9025         9035             0.01         1.00 +- 0.01         1.00 +- 0.01  [4 ] TO BT BT SC
                   c6ccd          8675         8640             0.07         1.00 +- 0.01         1.00 +- 0.01  [5 ] TO BT BT SC BT
                  cc6ccd          8446         8392             0.17         1.01 +- 0.01         0.99 +- 0.01  [6 ] TO BT BT SC BT BT
                 8cc6ccd          3370         3943            44.90         0.85 +- 0.01         1.17 +- 0.02  [7 ] TO BT BT SC BT BT SA




SA Opticks
------------

::

    410 
    411 
    412         command = propagate_to_boundary( p, s, rng );
    413         if(command == BREAK)    break ;           // BULK_ABSORB
    414         if(command == CONTINUE) continue ;        // BULK_REEMIT/BULK_SCATTER
    415         // PASS : survivors will go on to pick up one of the below flags, 
    416 
    417 
    418         if(s.optical.x > 0 )       // x/y/z/w:index/type/finish/value
    419         {
    420             command = propagate_at_surface(p, s, rng);
    421             if(command == BREAK)    break ;       // SURFACE_DETECT/SURFACE_ABSORB
    422             if(command == CONTINUE) continue ;    // SURFACE_DREFLECT/SURFACE_SREFLECT
    423         }
    424         else
    425         {
    426             //propagate_at_boundary(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    427             propagate_at_boundary_geant4_style(p, s, rng);     // BOUNDARY_RELECT/BOUNDARY_TRANSMIT
    428             // tacit CONTINUE
    429         }


    486 __device__ int
    487 propagate_at_surface(Photon &p, State &s, curandState &rng)
    488 {
    489 
    490     float u = curand_uniform(&rng);
    491 
    492     if( u < s.surface.y )   // absorb   
    493     {
    494         s.flag = SURFACE_ABSORB ;
    495         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    496         return BREAK ;
    ///
    ///         G4 doing this 20% more than Opticks
    ///
    497     }
    498     else if ( u < s.surface.y + s.surface.x )  // absorb + detect
    499     {
    500         s.flag = SURFACE_DETECT ;
    501         s.index.x = s.index.y ;   // kludge to get m2 into seqmat for BREAKERs
    502         return BREAK ;
    503     }
    504     else if (u  < s.surface.y + s.surface.x + s.surface.w )  // absorb + detect + reflect_diffuse 
    505     {
    506         s.flag = SURFACE_DREFLECT ;
    507         propagate_at_diffuse_reflector(p, s, rng);
    508         return CONTINUE;
    509     }
    510     else
    511     {
    512         s.flag = SURFACE_SREFLECT ;
    513         propagate_at_specular_reflector(p, s, rng );
    514         return CONTINUE;
    515     }
    516 }

::

     20 enum {
     21     OMAT,
     22     OSUR,
     23     ISUR,
     24     IMAT 
     25 };
     26 
     27 __device__ void fill_state( State& s, int boundary, uint4 identity, float wavelength )
     28 {       
     29     // boundary : 1 based code, signed by cos_theta of photon direction to outward geometric normal
     30     // >0 outward going photon
     31     // <0 inward going photon
     32     //  
     33     // NB the line is above the details of the payload (ie how many float4 per matsur) 
     34     //    it is just 
     35     //                boundaryIndex*4  + 0/1/2/3     for OMAT/OSUR/ISUR/IMAT 
     36     //      
     37             
     38     int line = boundary > 0 ? (boundary - 1)*BOUNDARY_NUM_MATSUR : (-boundary - 1)*BOUNDARY_NUM_MATSUR  ;
     39 
     40     // pick relevant lines depening on boundary sign, ie photon direction relative to normal
     41     // 
     42     int m1_line = boundary > 0 ? line + IMAT : line + OMAT ;
     43     int m2_line = boundary > 0 ? line + OMAT : line + IMAT ;
     44     int su_line = boundary > 0 ? line + ISUR : line + OSUR ;
     45     
     46     //  consider photons arriving at PMT cathode surface
     47     //  geometry normals are expected to be out of the PMT 
     48     //
     49     //  boundary sign will be -ve : so line+3 outer-surface is the relevant one
     50     
     51     s.material1 = boundary_lookup( wavelength, m1_line, 0);  
     52     s.material2 = boundary_lookup( wavelength, m2_line, 0);
     53     s.surface   = boundary_lookup( wavelength, su_line, 0);
     54     
     55     s.optical = optical_buffer[su_line] ;   // index/type/finish/value
     56     
     57     s.index.x = optical_buffer[m1_line].x ; // m1 index
     58     s.index.y = optical_buffer[m2_line].x ; // m2 index 
     59     s.index.z = optical_buffer[su_line].x ; // su index
     60     s.index.w = identity.w   ;
     61 
     62     s.identity = identity ;
     63 
     64 }



Check s.optical::


    ipython -i proplib.py 

    In [1]: op.shape
    Out[1]: (123, 4, 4)

    In [2]: op
    Out[2]: 
    array([[[ 13,   0,   0,   0],
            [  #0,   0,   0,   0],     # no OSUR
            [  #0,   0,   0,   0],     # no ISUR
            [ 13,   0,   0,   0]],

           [[ 13,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [ 12,   0,   0,   0]],

           [[ 12,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [ 15,   0,   0,   0]],

           ..., 
           [[  9,   0,   0,   0],
            [ 43,   0,   3, 100],     # has OSUR
            [  0,   0,   0,   0],
            [ 24,   0,   0,   0]],

           [[  8,   0,   0,   0],
            [ 44,   0,   3, 100],
            [  0,   0,   0,   0],
            [ 19,   0,   0,   0]],

           [[ 12,   0,   0,   0],
            [  0,   0,   0,   0],
            [  0,   0,   0,   0],
            [ 36,   0,   0,   0]]], dtype=uint32)


OMAT/IMAT lines just contain 1-based material indices::

    In [3]: op[:,0]  
    Out[3]: 
    array([[13,  0,  0,  0],
           [13,  0,  0,  0],
           [12,  0,  0,  0],
           [15,  0,  0,  0],
           [15,  0,  0,  0],
           [18,  0,  0,  0],
           [20,  0,  0,  0],
           ...

    In [4]: op[:,3]
    Out[4]: 
    array([[13,  0,  0,  0],
           [12,  0,  0,  0],
           [15,  0,  0,  0],
           [17,  0,  0,  0],
           [18,  0,  0,  0],
           [20,  0,  0,  0],
           [26,  0,  0,  0],
           [15,  0,  0,  0],


OSUR/ISUR lines contain surface info::

    In [5]: op[:,1]
    Out[5]: 
    array([[  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  1,   0,   3, 100],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           ...
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [ 12,   0,   3, 100],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [  0,   0,   0,   0],
           [ 13,   0,   3, 100],
           [  0,   0,   0,   0],
           [ 14,   0,   3, 100],
           [ 15,   0,   3, 100],
           [ 16,   0,   3, 100],
           [  0,   0,   0,   0],
           [ 17,   0,   3, 100],
           [ 18,   0,   3, 100],


::

    op --bnd

    2016-10-28 12:30:12.396 INFO  [347098] [GBndLib::dump@787] GBndLib::dump ni 123
     (  0) om:                   Vacuum os:                          is:                          im:                   Vacuum
     (  1) om:                   Vacuum os:                          is:                          im:                     Rock
     (  2) om:                     Rock os:                          is:                          im:                      Air
     (  3) om:                      Air os:     NearPoolCoverSurface is:                          im:                      PPE
     (  4) om:                      Air os:                          is:                          im:                Aluminium
     (  5) om:                Aluminium os:                          is:                          im:                     Foam
     (  6) om:                     Foam os:                          is:                          im:                 Bakelite
     (  7) om:                 Bakelite os:                          is:                          im:                      Air
     (  8) om:                      Air os:                          is:                          im:                   MixGas
     (  9) om:                      Air os:                          is:                          im:                      Air
     ( 10) om:                      Air os:                          is:                          im:                     Iron
     ( 11) om:                     Rock os:                          is:                          im:                     Rock
     ( 12) om:                     Rock os:                          is:                          im:                DeadWater
     ( 13) om:                DeadWater os:     NearDeadLinerSurface is:                          im:                    Tyvek
     ( 14) om:                    Tyvek os:                          is:      NearOWSLinerSurface im:                 OwsWater
     ( 15) om:                 OwsWater os:                          is:                          im:                    Tyvek
     ( 16) om:                    Tyvek os:                          is:    NearIWSCurtainSurface im:                 IwsWater
     ( 17) om:                 IwsWater os:                          is:                          im:                 IwsWater
     ( 18) om:                 IwsWater os:     SSTWaterSurfaceNear1 is:                          im:           StainlessSteel
     ( 19) om:           StainlessSteel os:                          is:            SSTOilSurface im:               MineralOil
     ( 20) om:               MineralOil os:                          is:                          im:                  Acrylic
     ( 21) om:                  Acrylic os:                          is:                          im:       LiquidScintillator
     ( 22) om:       LiquidScintillator os:                          is:                          im:                  Acrylic
     ( 23) om:                  Acrylic os:                          is:                          im:                GdDopedLS






G4 SA
--------

::


    232 #ifdef USE_CUSTOM_BOUNDARY
    233 unsigned int OpPointFlag(const G4StepPoint* point, const DsG4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
    234 #else
    235 unsigned int OpPointFlag(const G4StepPoint* point, const G4OpBoundaryProcessStatus bst, CStage::CStage_t stage)
    236 #endif
    237 {
    238     G4StepStatus status = point->GetStepStatus()  ;
    239     // TODO: cache the relevant process objects, so can just compare pointers ?
    240     const G4VProcess* process = point->GetProcessDefinedStep() ;
    241     const G4String& processName = process ? process->GetProcessName() : "NoProc" ;
    242 
    243     bool transportation = strcmp(processName,"Transportation") == 0 ;
    244     bool scatter = strcmp(processName, "OpRayleigh") == 0 ;
    245     bool absorption = strcmp(processName, "OpAbsorption") == 0 ;
    246 
    247     unsigned flag(0);
    248 
    249     if(absorption && status == fPostStepDoItProc )
    250     {
    251         flag = BULK_ABSORB ;
    252     }
    253     else if(scatter && status == fPostStepDoItProc )
    254     {
    255         flag = BULK_SCATTER ;
    256     }
    257     else if(transportation && status == fWorldBoundary )
    258     {
    259         flag = SURFACE_ABSORB ;   // kludge for fWorldBoundary - no surface handling yet 
    260     }
    261     else if(transportation && status == fGeomBoundary )
    262     {
    263         flag = OpBoundaryFlag(bst) ; // BOUNDARY_TRANSMIT/BOUNDARY_REFLECT/NAN_ABORT/SURFACE_ABSORB/SURFACE_DETECT/SURFACE_DREFLECT/SURFACE_SREFLECT
    264     }
    265     else if( stage == CStage::REJOIN )
    266     {
    267         flag = BULK_REEMIT ;
    268     }
    269     else
    270     {
    271         LOG(warning) << " OpPointFlag ZERO  "
    272                      << " proceesDefinedStep? " << processName
    273                      << " stage " << CStage::Label(stage)
    274                      ;
    275     }
    276     return flag ;
    277 }



    158 #ifdef USE_CUSTOM_BOUNDARY
    159 unsigned int OpBoundaryFlag(const DsG4OpBoundaryProcessStatus status)
    160 #else
    161 unsigned int OpBoundaryFlag(const G4OpBoundaryProcessStatus status)
    162 #endif
    163 {
    164     unsigned flag = 0 ;
    165     switch(status)
    166     {
    167         case FresnelRefraction:
    168         case SameMaterial:
    169                                flag=BOUNDARY_TRANSMIT;
    170                                break;
    171         case TotalInternalReflection:
    172         case       FresnelReflection:
    173                                flag=BOUNDARY_REFLECT;
    174                                break;
    175         case StepTooSmall:
    176                                flag=NAN_ABORT;
    177                                break;
    178         case Absorption:
    179                                flag=SURFACE_ABSORB ;
    180                                break;
    181         case Detection:
    182                                flag=SURFACE_DETECT ;
    183                                break;
    184         case SpikeReflection:
    185                                flag=SURFACE_SREFLECT ;
    186                                break;
    187         case LobeReflection:
    188         case LambertianReflection:
    189                                flag=SURFACE_DREFLECT ;
    190                                break;
    191         case Undefined:
    192         case BackScattering:
    193         case NotAtBoundary:
    194         case NoRINDEX:



::

    1093 void DsG4OpBoundaryProcess::DoAbsorption()
    1094 {
    1095     //LOG(info) << "DsG4OpBoundaryProcess::DoAbsorption"
    1096     //          << " theEfficiency " << theEfficiency
    1097     //          ; 
    1098 
    1099     theStatus = Absorption;
    1100 
    1101     if ( G4BooleanRand(theEfficiency) )
    1102     {
    1103         // EnergyDeposited =/= 0 means: photon has been detected
    1104         theStatus = Detection;
    1105         aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    1106     }
    1107     else
    1108     {
    1109         aParticleChange.ProposeLocalEnergyDeposit(0.0);
    1110     }
    1111 
    1112     NewMomentum = OldMomentum;
    1113     NewPolarization = OldPolarization;
    1114 
    1115 //  aParticleChange.ProposeEnergy(0.0);
    1116     aParticleChange.ProposeTrackStatus(fStopAndKill);
    1117 }


::

     704 void DsG4OpBoundaryProcess::DielectricMetal()
     705 {
     706         G4int n = 0;
     707 
     708     do {
     709 
     710            n++;
     711 
     712            if( !G4BooleanRand(theReflectivity) && n == 1 ) {
     713 
     714              // Comment out DoAbsorption and uncomment theStatus = Absorption;
     715              // if you wish to have Transmission instead of Absorption
     716 
     717              DoAbsorption();
     718              // theStatus = Absorption;
     719              break;
     720 
     721            }
     722            else {





