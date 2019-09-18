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

#include "SStr.hh"
#include "SSys.hh"
#include "BFile.hh"

#include "NLookup.hpp"
#include "NGLM.hpp"
#include "NPY.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "CGenstepCollector.hh"

#include "OPTICKS_LOG.hh"


unsigned mock_num_steps( unsigned e )
{
    unsigned ns = 0 ; 
    switch( e % 10 )
    {
       case 0: ns = 10 ; break ;
       case 1: ns = 50 ; break ;
       case 2: ns = 60 ; break ;
       case 3: ns = 80 ; break ;
       case 4: ns = 10 ; break ;
       case 5: ns = 100 ; break ;
       case 6: ns = 30 ; break ;
       case 7: ns = 300 ; break ;
       case 8: ns = 20 ; break ;
       case 9: ns = 10 ; break ;
   }
   return ns ;  
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    unsigned mock_num_evt = 10 ; 

    Opticks ok(argc, argv);
    OpticksHub hub(&ok);



    NLookup* lookup = hub.getLookup(); 

    if(ok.isLegacy())
    {
        lookup->close(); 
    }
    // For real genstep collection it is essential 
    // to close the lookup and do cross-referencing
    // which will need OpticksHub::overrideMaterialMapA
    // using the Geant4 material map, but for machinery testing
    // the lookup is not used, so can live without this setup.

    CGenstepCollector cc(lookup);
    NPY<float>* gs = cc.getGensteps(); 


    for(unsigned e=0 ; e < mock_num_evt ; e++)
    {
        unsigned num_steps = mock_num_steps(e); 

        for(unsigned i=0 ; i < num_steps ; i++) CGenstepCollector::Instance()->collectMachineryStep(e*1000+i);

        const char* path_ = SSys::fmt("$TMP/CGenstepCollectorTest%u.npy",e) ;
        std::string spath = BFile::preparePath(path_);
        const char* path = spath.c_str();  

        gs->save(path);
        gs->reset();

        SSys::npdump(path, "np.uint32" );

        ok.profile(SStr::Concat(NULL, e)) ; 
    }

    ok.dumpProfile(argv[0]); 
    ok.saveProfile();
   

    return ok.getRC() ; 
}


/*

In [1]: import sys, os, numpy as np ; np.set_printoptions(suppress=True, precision=3)

In [2]: a=np.load(os.path.expandvars("$TMP/CGenstepCollectorTest_vm.npy"))

In [3]: a.shape
Out[3]: (100,)

In [4]: a
Out[4]: 
array([  1.,  10.,  10.,  11.,  19.,  19.,  19.,  20.,  28.,  28.,  28.,
        28.,  28.,  28.,  30.,  38.,  38.,  38.,  38.,  38.,  38.,  39.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,  47.,
        47.,  47.,  47.,  48.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,  57.,
        57.], dtype=float32)

In [5]: import matplotlib.pyplot as plt 

In [6]: plt.plot(a)
Out[6]: [<matplotlib.lines.Line2D at 0x112f0bdd0>]

In [7]: plt.show()


*/



