#include "GAry.hh"
#include "GProperty.hh"
#include "assert.h"



void test_createReciprocalCDF()
{
    GProperty<float>* slow = GProperty<float>::load("/tmp/slowcomponent.npy");
    slow->Summary("slow",20);

    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);

    bool reciprocal_domain = true ; 
    GProperty<float>* rcdf = slow->createCDF(reciprocal_domain);
    rcdf->Summary("rcdf", 20);

    //  maxdiff (rcdf - pcdf)*1e9 :   596.0464
    float mx = GProperty<float>::maxdiff( rcdf, pcdf );
    assert(mx < 1e-6) ; 
    printf("maxdiff (rcdf - pcdf)*1e9 : %10.4f\n", mx*1e9 );

    GAry<float>* domdif = GAry<float>::subtract( rcdf->getDomain(), pcdf->getDomain() );
    domdif->Summary("domdif*1e6",20, 1e6); 

    GAry<float>* valdif = GAry<float>::subtract( rcdf->getValues(), pcdf->getValues() );
    valdif->Summary("valdif*1e6",20, 1e6); 
}



void test_sampling_0()
{
    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);

    GAry<float>* vcdf = pcdf->getValues();
    GAry<float>* ua = GAry<float>::urandom(10);

    for(unsigned int i=0 ; i < ua->getLength() ; i++)
    {
         float u = ua->getValue(i); 
         unsigned int a = vcdf->binary_search(u);
         unsigned int b = vcdf->sample_cdf(u);
         assert(a == b);
         //printf("i %u  u %15.6f a %u b %u \n", i, u, a, b );
    }
}

void test_sampling()
{
    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);
    pcdf->save("/tmp/test_sampling_save.npy");

    GAry<float>* psample = pcdf->sampleCDF(1e6); 
    psample->Summary("pcdf sample *1e3", 1, 1e3);
    psample->save("/tmp/psampleCDF.npy");

    /*
    Use ipython to plot the generated sample using::

       i
       a = np.load("/tmp/psampleCDF.npy")
       plt.ion()
       plt.hist(1/a, bins=100, log=True)

    */
}

void test_createInverseCDF()
{
    GProperty<float>* pcdf = GProperty<float>::load("/tmp/reemission_cdf.npy");  // 79.98 -> 799.89
    GAry<float>* psample = pcdf->sampleCDF(1e6); 
    psample->Summary("psample *1e3", 1, 1e3);
    psample->save("/tmp/psample.npy");

   /*
        psample = 1/np.load("/tmp/psample.npy")   199.9 -> 799
        plt.hist(psample, bins=50, log=True)

        Somehow this one avoids the problem, with clean turn on at ~199.9  

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


    GProperty<float>* slow = GProperty<float>::load("/tmp/slowcomponent.npy");   // 79.99 -> 799.898  odd zero bins at low end
    slow->Summary("slow",20);

   /*
     Odd zero bins may be implicated in  80..200 nm bizarreness  

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


    bool reciprocal_domain = true ; 
    GProperty<float>* rcdf = slow->createCDF(reciprocal_domain);
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


    GProperty<float>* icdf = rcdf->createInverseCDF(10001);
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
    */


    /*

         to compare

             ./GPropertyTest.sh


    */


}


int main(int argc, char** argv)
{
    //test_createReciprocalCDF();
    //test_sampling();
    test_createInverseCDF();
    return 0 ;
}
