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

// op --gscintillatorlib

#include "NPY.hpp"

#include "Opticks.hh"
#include "GPropertyMap.hh"
#include "GScintillatorLib.hh"


#include "OPTICKS_LOG.hh"
#include "GGEO_BODY.hh"



/*
In [1]: s = np.load("/usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GScintillatorLib.npy")

In [2]: s
Out[2]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.717],
        [ 207.546],
        [ 180.   ]]], dtype=float32)

In [3]: s.shape
Out[3]: (1, 4096, 1)



In [1]: a = np.load("/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GScintillatorLib/GScintillatorLib.npy")

In [2]: a
Out[2]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]],

       [[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)

In [3]: a.shape
Out[3]: (2, 4096, 1)



In [4]: b = np.load("$TMP/GScintillatorLib.npy")

In [5]: b.shape
Out[5]: (2, 4096, 1)

In [6]: b
Out[6]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]],

       [[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)


In [7]: c = np.load("$TMP/GScintillatorLib0.npy")

In [8]: c
Out[8]: 
array([[[ 800.   ],
        [ 698.82 ],
        [ 661.96 ],
        ..., 
        [ 215.715],
        [ 207.545],
        [ 180.   ]]], dtype=float32)

In [9]: c.shape
Out[9]: (1, 4096, 1)





 45 2016-07-06 17:58:35.348 INFO  [12948] [OpEngine::prepareOptiX@112] OpEngine::prepareOptiX (OColors)
 46 2016-07-06 17:58:35.348 INFO  [12948] [OpEngine::prepareOptiX@119] OpEngine::prepareOptiX (OSourceLib)
 47 2016-07-06 17:58:35.349 INFO  [12948] [OpEngine::prepareOptiX@124] OpEngine::prepareOptiX (OScintillatorLib)
 48 2016-07-06 17:58:35.349 INFO  [12948] [OScintillatorLib::makeReemissionTexture@39] OScintillatorLib::makeReemissionTexture  nx 4096 ny 1 ni 2 nj 4096 nk 1 step 0.000244141 empty 0
 49 *** Error in `/home/ihep/simon-dev-env/env-dev-2016july4/local/opticks/lib/OpEngineTest': free(): invalid next size (fast): 0x0000000002584420 ***
 50 ======= Backtrace: =========




*/


const char* TMPDIR = "$TMP/ggeo/GScintillatorLibTest" ; 


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc,argv);


    Opticks* ok = new Opticks(argc, argv);
    ok->configure();

    GScintillatorLib* slib = GScintillatorLib::load(ok);
    slib->dump();

    //const char* name = "LiquidScintillator" ;
    const char* name = "LS" ;

    GPropertyMap<float>* ls = slib->getRaw(name);
    LOG(info) << " ls " << ls ; 
    if(ls)
    {
        ls->dump("ls");
    } 
    else
    {
         LOG(error) << " FAILED TO FIND " << name ;
         return 0 ; 
    }



    NPY<float>* buf = slib->getBuffer();
    buf->Summary();
    const char* name_ = "GScintillatorLib.npy" ;

    LOG(info) << " save GScintillatorLib buf  "
              << " to dir  " << TMPDIR
              << " name_ " << name_
              << " shape " << buf->getShapeString()
              ; 

    buf->save(TMPDIR, name_);


    NPY<float>* buf0 = buf->make_slice("0:1") ;
    const char* name0 = "GScintillatorLib0.npy" ;

    LOG(info) << " save GScintillatorLib buf0  "
              << " to dir " << TMPDIR
              << " name0 " << name0
              << " shape " << buf0->getShapeString()
              ; 

    buf0->save(TMPDIR, name0);

    


    return 0 ;
}

