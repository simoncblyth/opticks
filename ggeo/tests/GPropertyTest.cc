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
#include <cstdio>

#include "BOpticksResource.hh"


#include "GDomain.hh"
#include "GAry.hh"
#include "GProperty.hh"


#include "OPTICKS_LOG.hh"

/*
// Why does lookup "sampling" require so many more bins to get good results than standard sampling ?
//
// * maybe because "good" means it matches a prior standard sampling and in
//   the limit of many bins the techniques converge ?
//   **Nope**
//
// * A simpler (and thus more believable) explanation is that domain lookup 
//   is using a fixed raster on 0:1 to allow jumping to correct value in the 0:1 domain based 
//   on a uniform random throw. This mimicks how GPU texture lookups are used.
//   This means have to have loadsa values all the way from 0->1 even though 
//   the action is in small portions of the range.  
//
//   Conversely binary search of CDF values doesnt need the regular
//   raster, as the values are disposed according to the numerical integration
//   kinda like a variable raster arranged precisely as appropriate.  Which means
//   that binary bin search can work well with as low as ~50 domain values.
//   Lookup sampling needs ~100x bins to match.  
// 
//   Of course the advantage of lookup sampling is can jump straight to the generated
//   value with no binary searching.   
//
//

    * not very sure of my terminology 

    * http://en.wikipedia.org/wiki/Cumulative_distribution_function
    * http://en.wikipedia.org/wiki/Quantile_function
    * http://en.wikipedia.org/wiki/Inverse_transform_sampling
    * http://www.pbrt.org/chapters/pbrt_chapter7.pdf

      * The inverse of the cdf is called the quantile function.


*/


typedef GAry<float> A ;
typedef GProperty<float> P ;
typedef GDomain<float> D ;


void test_loadRefractiveIndex()
{
   P* ri = P::load("$OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/glass/schott/F2.npy");
   if(ri == NULL)
   {
       LOG(error) << " load failed " ;
       return ;  
   } 

   ri->Summary("F2 ri", 100);
}

void test_planck()
{
    D* dom = new D(300.,800.,1.); 
    P* p = P::planck_spectral_radiance(dom);
    p->save("$TMP/dom_planck.npy"); 
}


void test_createSliced()
{
    P* slow = P::load("$TMP/slowcomponent.npy");
    //slow->Summary("slow",20);

    P* sslow = slow->createSliced(2, slow->getLength());
    //sslow->Summary("sslow", 20);

    assert(sslow->getLength() == slow->getLength() - 2);
}


void test_createReciprocalCDF()
{
    P* pcdf = P::load("$TMP/reemission_cdf.npy");
    pcdf->Summary("pcdf",20);

    P* slow = P::load("$TMP/slowcomponent.npy");
    slow->Summary("slow",20);

    P* rrd = slow->createReversedReciprocalDomain();
    P* rcdf = rrd->createCDF();
    rcdf->Summary("rcdf", 20);

    //  maxdiff (rcdf - pcdf)*1e9 :   596.0464
    float mx = P::maxdiff( rcdf, pcdf );
    printf("maxdiff (rcdf - pcdf)*1e9 : %10.4f\n", mx*1e9 );

    if(mx > 1e-6)
    {
        GAry<float>* domdif = GAry<float>::subtract( rcdf->getDomain(), pcdf->getDomain() );
        domdif->Summary("domdif*1e6",20, 1e6); 

        GAry<float>* valdif = GAry<float>::subtract( rcdf->getValues(), pcdf->getValues() );
        valdif->Summary("valdif*1e6",20, 1e6); 
    }

    assert(mx < 1e-6) ; 
}


void test_traditional_remission_cdf_sampling()
{
    P* pcdf = P::load("$TMP/reemission_cdf.npy");  // 79.98 -> 799.89
    A* psample = pcdf->sampleCDF(1000000); 
    psample->Summary("psample *1e3", 1, 1e3);
    psample->save("$TMP/psample.npy");

   /*
        psample = 1/np.load("$TMP/psample.npy")   199.9 -> 799
        plt.ion()
        plt.hist(psample, bins=50, log=True)

        Somehow this one avoids the zero bin problem, with clean turn on at ~199.9 nm
        (presumably a detail of the sampling implementation?)

        pcdf = np.load("$TMP/reemission_cdf.npy")
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


void test_inverseCDF_lookup()
{
    bool reciprocate = true ; 

    P* slow = P::load("$TMP/slowcomponent.npy");        // 79.99 -> 799.898  three zero values at low nm end

    P* rrd = reciprocate ? slow->createReversedReciprocalDomain() : slow ;   

    P* srrd= rrd->createZeroTrimmed();                  // trim extraneous zero values, leaving at most one zero at either extremity

    if(reciprocate)
    {
        assert( srrd->getLength() == rrd->getLength() - 2); // trims 2 values
    }
    else
    {
        assert( srrd->getLength() == rrd->getLength() - 1); // why the difference, reversal and reciprocation should not touch the zero values ? 
    }

     //
     // "have to used reciprocal "energywise" domain for G4/NuWa agreement"
     //
     //     this statement has been checked with both traditional CDF sampling
     //     and inverse CDF lookups
     //     
     //     

    P* rcdf = srrd->createCDF();


    //unsigned int nicdf = 101 ; //   unacceptably bad
    //unsigned int nicdf = 1001 ; //  visible discrep in 550-650 nm
    //unsigned int nicdf = 1024 ; //  visible discrep in 550-650 nm
    //unsigned int nicdf = 2048 ; //  still visible
      unsigned int nicdf = 4096 ; //  minor discrep ~600nm
    //unsigned int nicdf = 8192 ; //  indistinguishable from standard sampling  
    //unsigned int nicdf = 10001 ; // indistinguishable from standard sampling 

    P* icdf = rcdf->createInverseCDF(nicdf); 

    icdf->getValues()->reciprocate();

    icdf->save("$TMP/icdf.npy");

    // two ways yield same characteristics
      A* isample = icdf->lookupCDF(1000000);
    //A* isample = icdf->lookupCDF_ValueLookup(1e6);

    isample->Summary("icdf->lookupCDF(1e6)  *1e3", 1, 1e3);

    isample->save("$TMP/isample.npy");
}


void test_GROUPVEL()
{
    P* ri = P::load("$OPTICKS_INSTALL_PREFIX/opticksaux/refractiveindex/tmp/glass/schott/F2.npy");
    ri->Summary("ri(nm)");
   
    P* vg = GProperty<float>::make_GROUPVEL(ri); 
    vg->Summary("vg");


}




int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    BOpticksResource* rsc = BOpticksResource::Get(NULL) ;   // sets OPTICKS_INSTALL_PREFIX envvar
    assert(rsc); 
  
    //test_createSliced();
    //test_createReciprocalCDF();
    //test_traditional_remission_cdf_sampling();
    //test_inverseCDF_lookup();

    //test_loadRefractiveIndex();
    //test_planck();

    test_GROUPVEL();

    return 0 ;
}
