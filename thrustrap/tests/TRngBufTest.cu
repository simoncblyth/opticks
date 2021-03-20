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


#include <iostream> 
#include <iomanip> 



#include "SSys.hh"
#include "SStr.hh"
#include "BFile.hh"
#include "NPY.hpp"

#include "THRAP_HEAD.hh"
#include <thrust/device_vector.h>
#include "TRngBuf.hh"
#include "TUtil.hh"
#include "THRAP_TAIL.hh"

#include "OPTICKS_LOG.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << argv[0] ;

    //static const unsigned NI_DEFAULT = 100000 ; 
    static const unsigned NI_DEFAULT = 10000 ;    // decrease default from 100k to 10k, see notes/issues/longer-thrap-tests-flakey-on-macOS.rst 
    static const unsigned IBASE = SSys::getenvint("TRngBuf_IBASE", 0) ; 
    static const unsigned NI = SSys::getenvint("TRngBuf_NI", NI_DEFAULT ); 
    static const unsigned NJ = 16 ; 
    static const unsigned NK = 16 ; 

    bool default_ni = NI == NI_DEFAULT ;  


    NPY<double>* ox = NPY<double>::make(NI, NJ, NK);

    ox->zero();

    thrust::device_vector<double> dox(NI*NJ*NK);

    CBufSpec spec = make_bufspec<double>(dox); 

    TRngBuf<double> trb(NI, NJ*NK, spec );

    trb.setIBase(IBASE) ; 

    trb.generate(); 

    trb.download<double>(ox, true) ; 


    const char* path = default_ni ? 
                                     SStr::Concat("$TMP/TRngBufTest_", IBASE, ".npy") 
                                  :
                                     SStr::Concat("$TMP/TRngBufTest_", IBASE, "_", NI, ".npy") 
                                  ; 


    LOG(info) << " save " << path ; 

    ox->save(path)  ;

    std::string spath = BFile::FormPath(path); 

    SSys::npdump(spath.c_str(), "np.float64", NULL, "suppress=True,precision=8" );

    cudaDeviceSynchronize();  
}


