/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <cassert>

#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"

#include "PLOG.hh"
#include "GGEO_LOG.hh"


typedef GProperty<float> P ; 


int test_createInverseCDF_Debug()
{

    const char* path = "$TMP/slowcomponent.npy" ;

    P* slow = P::load(path);   // 79.99 -> 799.898  odd zero bins at low end

    if(!slow)
    {
        LOG(fatal) << "FAILED TO LOAD " << path ;
        return 0 ; 
    }



    slow->Summary("slow",20);

   /*
     Odd zero bins may be implicated in  80..200 nm bizarreness  
     YEP, trimming the zero bins allows to create an InverseCDF that yields
     via lookup an agreeable sample


        In [31]: np.set_printoptions(precision=16)

        In [32]: s
        Out[32]: 
        array([[  79.989837646484375 ,    0.                ],
               [ 120.023468017578125 ,    0.                ],
               [ 199.9745941162109375,    0.                ],
               [ 329.999847412109375 ,    0.0060729999095201],
               [ 330.999786376953125 ,    0.0056940000504255],
               [ 332.000457763671875 ,    0.0051750000566244],

   */


    P* rrd = slow->createReversedReciprocalDomain();
    assert(rrd);
    rrd->Summary("rrd", 20);  
    rrd->save("$TMP/rrd.npy");        

   /*

    Leading zeros become trailing zeroes... 

    In [10]: rrd[:5]
    Out[10]: 
    array([[ 0.00125016,  0.        ],
           [ 0.00166666,  0.001787  ],
           [ 0.00166945,  0.001729  ],
           [ 0.00167224,  0.001969  ],
           [ 0.00167504,  0.002015  ]], dtype=float32)

    In [11]: rrd[-5:]
    Out[11]: 
    array([[ 0.00302115,  0.005694  ],
           [ 0.0030303 ,  0.006073  ],
           [ 0.00500064,  0.        ],
           [ 0.0083317 ,  0.        ],
           [ 0.01250159,  0.        ]], dtype=float32)


        plt.plot(1/rrd[:,0], rrd[:,1])      silly top hat 

   */

    //P* srrd = rrd->createSliced(0, -2);  // trim the trailing 2 zero bins
    P* srrd = rrd->createZeroTrimmed();  // trim the trailing 2 zero bins
    assert(srrd);
    srrd->Summary("srrd", 20);  
    srrd->save("$TMP/srrd.npy");        
    assert( srrd->getLength() == rrd->getLength() - 2);



    P* rcdf = srrd->createCDF();
    assert(rcdf);
    rcdf->Summary("rcdf", 20);
    rcdf->save("$TMP/rcdf.npy");

    /*
           rcdf = np.load("$TMP/rcdf.npy")
           plt.plot(1/rcdf[:,0],rcdf[:,1])     looks identical to pcdf


    In [18]: rcdf[-6:]
    Out[18]: 
    array([[ 0.0030120441224426,  0.9815963506698608],
           [ 0.0030211498960853,  0.9817459583282471],
           [ 0.0030303043313324,  0.9819088578224182],
           [ 0.0050006350502372,  1.                ],
           [ 0.008331703953445 ,  1.                ],
           [ 0.0125015880912542,  1.                ]], dtype=float32)

       Three zero bins yield three ones

    */


    P* icdf = rcdf->createInverseCDF(10001);  // +1 for nicer bin widths
    assert(icdf);
    icdf->save("$TMP/icdf.npy");

    /*
        icdf = np.load("$TMP/icdf.npy")
        plt.plot(icdf[:,0],1/icdf[:,1])       // looks like expected for inverted CDF

      600 nm
               \
                \_______________
                                \
                                 \
      200 nm                     |
               0                 1

        problem with handling the region near random throw 1 ? 

In [8]: np.set_printoptions(precision=16)

In [11]: icdf = np.load("$TMP/icdf.npy")

In [12]: icdf    ## use createInverseCDF(1001) for nicer bin widths
Out[12]: 
array([[ 0.                ,  0.0012501587625593],
       [ 0.0010000000474975,  0.0016202869592234],
       [ 0.0020000000949949,  0.0017872895114124],
       ..., 
       [ 0.9980000257492065,  0.0047828452661633],
       [ 0.999000072479248 ,  0.0048917401582003],
       [ 1.                ,  0.0125015880912542]], dtype=float32)


In [14]: 1/icdf[:,1]
Out[14]: 
array([ 799.89837646484375  ,  617.17462158203125  ,  559.5064697265625   ,
       ...,  209.08056640625     ,  204.4262237548828125,
         79.989837646484375 ], dtype=float32)


    */


    GAry<float>* isample = icdf->lookupCDF(1000000);
    assert(isample);
    isample->Summary("icdf->lookupCDF(1e6)  *1e3", 1, 1e3);
    isample->save("$TMP/isample.npy");
  
    /*

      In [1]: isample = 1/np.load("$TMP/isample.npy")     80:799
      In [3]: plt.hist(isample, bins=50, log=True)

       something funny on the low wavelength side  : between 80:200 nm
       seems to be a domain mismatch, some start from 80 and some from 200 ?

       Trimming 2 extreme zero bins fixes this,

    */
    return 0 ; 
}

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    GGEO_LOG_ ; 

    return test_createInverseCDF_Debug();
}
