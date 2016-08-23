Odd Photon Flag History ?
===========================

Issue
-------

* dayabay torch run has no reemission ? but running cerenkov does ... why ?

::

   histype.py --det dayabay --tag 1 --src torch 


DEFERRED
----------

Defer investigating this until can pose a more definite question: 
eg difference between G4 and Opticks flag histories.




TODO: migrate dscale
----------------------

* move cfg4/CPropLib.cc dscale machinery where it can be used for all property dumping 

::

    simon:cu blyth$ op --mat 0
    200 -rwxr-xr-x  1 blyth  staff  101856 Aug 19 14:09 /usr/local/opticks/lib/GMaterialLibTest
    proceeding : /usr/local/opticks/lib/GMaterialLibTest --mat 0
    2016-08-19 20:40:37.129 INFO  [225834] [main@88]  ok 
    2016-08-19 20:40:37.132 INFO  [225834] [main@92]  after load 
    F2 ri : b0ad5d685c9b6bfb9cbcb3d68e3a3024 : 101 
    d     320.000   2500.000
    v       1.696      1.582
    2016-08-19 20:40:37.132 INFO  [225834] [GMaterialLib::Summary@108] dump NumMaterials 39 NumFloat4 2
    2016-08-19 20:40:37.133 WARN  [225834] [GMaterialLib::dump@492] GPropertyMap<T>::  0       material m:GdDopedLS k:refractive_index absorption_length scattering_length reemission_prob group_velocity extra_y extra_z extra_w GdDopedLS
    2016-08-19 20:40:37.133 INFO  [225834] [GMaterialLib::dump@494]               domain    refractive_index   absorption_length   scattering_length     reemission_prob      group_velocity
               0.0166667              1.4536               0.001                 850                 0.4                 300
                  0.0125              1.4536               0.001                 850                 0.4                 300
                    0.01              1.4536               0.001                 850                 0.4                 300
              0.00833333              1.4536               0.001                 850                 0.4                 300
              0.00714286             1.66438               0.001                 850                 0.4                 300
                 0.00625             1.79252               0.001                 850             0.40001                 300
              0.00555556             1.52723               0.001                 850            0.410011                 300



cu/propagate.h
---------------

Validations of absorb/reemit/scatter splits against G4 in presense of scintillators not yet done...

::

    057 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     58 {
     59     float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index
     60     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     61     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     //
     //    Role the die twice to pick distances according to exponential probability distributions
     //    of absorbing or scattering, process with smaller distance wins
     //    so long as the geometry distance to boundary doesnt trump that.
     // 
     //    Role again within the absorption branch to decide based on reemission probability
     //    whether to reemit.    
     //
     62 
     63     if (absorption_distance <= scattering_distance)
     64     {
     65         if (absorption_distance <= s.distance_to_boundary)
     66         {
     67             p.time += absorption_distance/speed ;
     68             p.position += absorption_distance*p.direction;
     69 
     70             float uniform_sample_reemit = curand_uniform(&rng);
     71             if (uniform_sample_reemit < s.material1.w)                       // .w:reemission_prob
     72             {
     73                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
     74                 // non-scintillators have zero reemission_prob
     75                 p.wavelength = reemission_lookup(curand_uniform(&rng));
     76                 p.direction = uniform_sphere(&rng);
     77                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
     78                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
     79                 //p.flags.i.w |= BULK_REEMIT;
     80                 s.flag = BULK_REEMIT ;
     81                 return CONTINUE;
     82             }
     83             else
     84             {
     85                 //p.flags.i.w |= BULK_ABSORB;
     86                 s.flag = BULK_ABSORB ;
     87                 return BREAK;
     88             }
     89         }
     90         //  otherwise sail to boundary  
     91     }
     92     else
     93     {
     94         if (scattering_distance <= s.distance_to_boundary)
     95         {
     96             p.time += scattering_distance/speed ;
     97             p.position += scattering_distance*p.direction;
     98 
     99             rayleigh_scatter(p, rng);
    100 
    101             //p.flags.i.w |= RAYLEIGH_SCATTER;
    102             s.flag = BULK_SCATTER;
    103             p.flags.i.x = 0 ;  // no-boundary-yet for new direction
    104 
    105             return CONTINUE;
    106         }
    107         //  otherwise sail to boundary 



Issue Where is the Reemission ?
--------------------------------

So much SC no RE looks very wrong::

    simon:ana blyth$ histype.py --det dayabay --tag 1 --src torch 
    histype.py --det dayabay --tag 1 --src torch
    [2016-08-19 16:56:02,581] p24445 {./histype.py:55} INFO - loaded ph from /tmp/blyth/opticks/evt/dayabay/torch/1/ph.npy shape (100000, 1, 2) 
         14076 TO SC SC SC SC SC SC SC SC SC 
          8432 TO AB 
          7202 TO SC AB 
          6965 TO SA 
          6512 TO SC SC SA 
          6345 TO SC SC SC SA 
          6118 TO SC SA 
          5899 TO SC SC AB 
          5623 TO SC SC SC SC SA 
          5055 TO SC SC SC SC SC SA 
          4887 TO SC SC SC AB 
          4257 TO SC SC SC SC SC SC SA 
          3787 TO SC SC SC SC AB 
          3376 TO SC SC SC SC SC SC SC SA 
          2978 TO SC SC SC SC SC AB 
          2803 TO SC SC SC SC SC SC SC SC SA 
          2411 TO SC SC SC SC SC SC AB 
          1847 TO SC SC SC SC SC SC SC AB 
          1380 TO SC SC SC SC SC SC SC SC AB 
             4 TO SC BT BT SC SC SC SC SC SC 
             3 TO SC SC SC BT BT SC SC SC SC 
             2 TO SC SC BT BT SC SC SC SC SC 
             2 TO SC SC BT BT SA 
             2 TO SC SC SC SC SC BT BT SC SC 
             2 TO SC SC SC BT BT SA 
             2 TO SC SC SC SC SC BT BT BT BT 
             2 TO SC SC SC BT BT SC SA 
             1 TO SC BT BT SC SC SC SA 
             1 TO SC SC SC BT BT BT BT SC SC 
             1 TO SC SC SC SC SC SC SC SC BT 
             1 TO SC SC BT BT BT BR BR BR BR 
             1 TO SC SC SC SC SC SC BT BT SA 
             1 TO SC SC SC SC SC BT BT SC SA 
             1 TO SC BT AB 
             1 TO SC SC SC SC SC BT BT SC AB 
             1 TO SC SC SC SC SC BT BT BT SC 
             1 TO SC SC BT AB 
             1 TO SC BT BT AB 
             1 TO SC SC SC SC SC SC BT BT SC 
             1 TO SC BT BT SA 
             1 TO SC SC SC SC BT BT SC SA 
             1 TO SC BT BT SC SA 
             1 TO SC SC SC SC SC SC BT BT BT 
             1 TO SC BT BT SC SC SC SC SA 
             1 TO SC SC SC BT BT BT BT AB 
             1 TO SC BT BT SC SC AB 
             1 TO SC SC SC BT BT AB 
             1 TO SC SC SC SC BT BT SC AB 
    [2016-08-19 16:56:02,597] p24445 {/Users/blyth/opticks/ana/seq.py:28} WARNING - code bad abbr [?0?] s [TO SC SC BT BT ?0? BT BT] 
    [2016-08-19 16:56:02,597] p24445 {/Users/blyth/opticks/ana/seq.py:32} WARNING - code sees 1 bad abbr in [TO SC SC BT BT ?0? BT BT] 
             1 TO SC SC BT BT ?0? BT BT 
             1 TO SC SC SC SC BT BT SA 
             1 TO SC SC SC BT BT SC AB 
             1 TO SC SC SC SC SC BT AB 
             1 TO SC SC BT BT BT BT AB 
             1 TO SC BT BT BT BT BT AB 
             1 TO SC BT BT BT BT SA 



The PMT is in mineral oil so no RE is expected::

    simon:ana blyth$ ./histype.py --det PmtInBox --tag 10 --src torch 
    ./histype.py --det PmtInBox --tag 10 --src torch
    [2016-08-19 16:59:02,357] p24463 {./histype.py:55} INFO - loaded ph from /tmp/blyth/opticks/evt/PmtInBox/torch/10/ph.npy shape (100000, 1, 2) 
         67948 TO BT SA 
         21648 TO BT SD 
          4581 TO BT BT SA 
          3794 TO AB 
           640 TO SC SA 
           444 TO BT AB 
           350 TO BT BT AB 
           283 TO BR SA 
            81 TO SC BT SA 
            51 TO BT BT SC SA 
            40 TO SC AB 
            36 TO BT BR BR BT SA 
            28 TO BR AB 
            20 TO SC BT SD 
             9 TO BT BT SC BT BR BT SA 
             8 TO SC SC SA 
             7 TO SC BT BT SA 
             6 TO BR SC SA 
             4 TO BT BR BR BR BR BT BT BR BT 
             4 TO BT BR BR BT AB 
             3 TO SC BR SA 
             2 TO BT BT SC BT BT BT BT BT SA 
             2 TO BT BT SC BT BR BT AB 
             2 TO SC SC BT SA 
             1 TO BT BT SC BT BT BT BT BT BT 
             1 TO BT BR AB 
             1 TO BT BT SC BT BT BT BR BT BT 
             1 TO BT BR BR AB 
             1 TO SC BT BT AB 
             1 TO BT BT SC BT BT AB 
             1 TO BR SC BT BR BT SA 
             1 TO BT BT SC BT BT BR BR BR BR 
             1 TO BT BT SC AB 
    8cbbbcd TO BT BR BR BR BT SA 8cbbbcd 



