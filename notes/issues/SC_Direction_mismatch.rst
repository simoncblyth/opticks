SC_Direction_mismatch
=======================


AFTER LOG DOUBLE FIX SC POSITIONS MATCHING, BUT NOT THE SCATTER DIRECTION
---------------------------------------------------------------------------

See :doc:`AB_SC_Position_Time_mismatch`


::

    tboolean-;tboolean-box-ip


    In [10]: ab.aselhis = "TO SC SA"

    In [11]: ab.a.rpost()


    In [15]: ab.a.rpost() - ab.b.rpost()
    Out[15]: 
    A()sliced
    A([[[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [-340.3524,  167.8777,    0.    ,   -0.6293]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 235.5132, -415.3379,    0.    ,   -0.0098]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [  27.8305, -734.549 ,  332.1354,    0.4804]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 770.1423,    4.0879, -247.8869,   -0.1379]],

           [[   0.    ,    0.    ,    0.    ,    0.    ],
            [   0.    ,    0.    ,    0.    ,    0.    ],
            [ 668.9505,  525.8065,   -1.2387,    0.2643]],


    In [17]: ab.a.dindex("TO SC SA")
    Out[17]: '--dindex=420,595,1198,2658,5113,6058,10409,13143,13162,14510'

    In [18]: ab.b.dindex("TO SC SA")
    Out[18]: '--dindex=420,1198,2658,5113,6058,10409,13143,13162,17035,26237'





Following AB decision there is a reemission throw for which there is no G4 equivalent.
But its the end of the line for that RNG sub-seq so this will have no effect 
so long as not in scintillator.

::

    085     if (absorption_distance <= scattering_distance)
     86     {
     87         if (absorption_distance <= s.distance_to_boundary)
     88         {
     89             p.time += absorption_distance/speed ;
     90             p.position += absorption_distance*p.direction;
     91 
     92             float uniform_sample_reemit = curand_uniform(&rng);
     93             if (uniform_sample_reemit < s.material1.w)                       // .w:reemission_prob
     94             {
     95                 // no materialIndex input to reemission_lookup as both scintillators share same CDF 
     96                 // non-scintillators have zero reemission_prob
     97                 p.wavelength = reemission_lookup(curand_uniform(&rng));
     98                 p.direction = uniform_sphere(&rng);
     99                 p.polarization = normalize(cross(uniform_sphere(&rng), p.direction));
    100                 p.flags.i.x = 0 ;   // no-boundary-yet for new direction
    101 
    102                 s.flag = BULK_REEMIT ;
    103                 return CONTINUE;
    104             }
    105             else
    106             {
    107                 s.flag = BULK_ABSORB ;
    108                 return BREAK;
    109             }
    110         }
    111         //  otherwise sail to boundary  
    112     }
    113     else
    114     {
    115         if (scattering_distance <= s.distance_to_boundary)
    116         {
    117             p.time += scattering_distance/speed ;
    118             p.position += scattering_distance*p.direction;
    119 
    120             rayleigh_scatter(p, rng);
    121 



