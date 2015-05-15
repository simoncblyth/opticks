#include "GAry.hh"
#include "GProperty.hh"
#include "assert.h"

/*

   TODO: 

      * tidy up these tests, to make them more comprehensible

      * vary inverseCDF length, to see how few bins can get away with 

      * incorporate inverseCDF into GSubstanceLib texture prep
        (may need to make separate texture if the standard domain 
        length is insufficient) 

      * test on GPU generation 

*/


typedef GAry<float> A ;
typedef GProperty<float> P ;


void test_createSliced()
{
    P* slow = P::load("/tmp/slowcomponent.npy");
    slow->Summary("slow",20);

    P* sslow = slow->createSliced(2, slow->getLength());
    sslow->Summary("sslow", 20);

    assert(sslow->getLength() == slow->getLength() - 2);
}


void test_createReciprocalCDF()
{
    P* pcdf = P::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);



    P* slow = P::load("/tmp/slowcomponent.npy");
    slow->Summary("slow",20);

    P* rrd = slow->createReversedReciprocalDomain();
    P* rcdf = rrd->createCDF();
    rcdf->Summary("rcdf", 20);

    //  maxdiff (rcdf - pcdf)*1e9 :   596.0464
    float mx = P::maxdiff( rcdf, pcdf );
    assert(mx < 1e-6) ; 
    printf("maxdiff (rcdf - pcdf)*1e9 : %10.4f\n", mx*1e9 );

    GAry<float>* domdif = GAry<float>::subtract( rcdf->getDomain(), pcdf->getDomain() );
    domdif->Summary("domdif*1e6",20, 1e6); 

    GAry<float>* valdif = GAry<float>::subtract( rcdf->getValues(), pcdf->getValues() );
    valdif->Summary("valdif*1e6",20, 1e6); 
}


void test_traditional_remission_cdf_sampling()
{
    P* pcdf = P::load("/tmp/reemission_cdf.npy");  // 79.98 -> 799.89
    A* psample = pcdf->sampleCDF(1e6); 
    psample->Summary("psample *1e3", 1, 1e3);
    psample->save("/tmp/psample.npy");

   /*
        psample = 1/np.load("/tmp/psample.npy")   199.9 -> 799
        plt.ion()
        plt.hist(psample, bins=50, log=True)

        Somehow this one avoids the zero bin problem, with clean turn on at ~199.9 nm
        (presumably a detail of the sampling implementation?)

        pcdf = np.load("/tmp/reemission_cdf.npy")
        plt.plot(1/pcdf[:,0],pcdf[:,1])     

        Looks to be 1 all way out to 200nm

               ----.  
                    \
                     \
                      \
                       ----

    In [21]: pcdf[-6:]
    Out[21]: 
    array([[ 0.003012043889612 ,  0.9815963506698608],
           [ 0.0030211498960853,  0.9817460179328918],
           [ 0.0030303043313324,  0.9819088578224182],
           [ 0.0050006355158985,  1.                ],
           [ 0.008331703953445 ,  1.                ],
           [ 0.0125015880912542,  1.                ]], dtype=float32)


   */
}


void test_createInverseCDF()
{
    P* slow = P::load("/tmp/slowcomponent.npy");        // 79.99 -> 799.898  three zero values at low nm end

    P* rrd = slow->createReversedReciprocalDomain();    // have to used reciprocal "energywise" domain for G4/NuWa agreement

    P* srrd = rrd->createZeroTrimmed();                 // trim extraneous zero values, leaving at most one zero at either extremity

    assert( srrd->getLength() == rrd->getLength() - 2); // expect to trim 2 values

    P* rcdf = srrd->createCDF();

    P* icdf = rcdf->createInverseCDF(10001);

    icdf->save("/tmp/icdf.npy");

    A* isample = icdf->lookupCDF(1e6);

    isample->Summary("icdf->lookupCDF(1e6)  *1e3", 1, 1e3);

    isample->save("/tmp/isample.npy");
}



void test_createInverseCDF_Debug()
{

    P* slow = P::load("/tmp/slowcomponent.npy");   // 79.99 -> 799.898  odd zero bins at low end
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
    rrd->Summary("rrd", 20);  
    rrd->save("/tmp/rrd.npy");        

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
    srrd->Summary("srrd", 20);  
    srrd->save("/tmp/srrd.npy");        
    assert( srrd->getLength() == rrd->getLength() - 2);



    P* rcdf = srrd->createCDF();
    rcdf->Summary("rcdf", 20);
    rcdf->save("/tmp/rcdf.npy");

    /*
           rcdf = np.load("/tmp/rcdf.npy")
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


    P* icdf = rcdf->createInverseCDF(10001);
    icdf->save("/tmp/icdf.npy");

    /*
        icdf = np.load("/tmp/icdf.npy")
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

In [11]: icdf = np.load("/tmp/icdf.npy")

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


    GAry<float>* isample = icdf->lookupCDF(1e6);
    isample->Summary("icdf->lookupCDF(1e6)  *1e3", 1, 1e3);
    isample->save("/tmp/isample.npy");
  
    /*

      In [1]: isample = 1/np.load("/tmp/isample.npy")     80:799
      In [3]: plt.hist(isample, bins=50, log=True)

       something funny on the low wavelength side  : between 80:200 nm
       seems to be a domain mismatch, some start from 80 and some from 200 ?

       Trimming 2 extreme zero bins fixes this,

    */


    /*

         to compare

             ./GPropertyTest.sh


    */


}


int main(int argc, char** argv)
{
    //test_createSliced();
    //test_createReciprocalCDF();
    //test_sampling();

    test_traditional_remission_cdf_sampling();
    test_createInverseCDF();
    return 0 ;
}
