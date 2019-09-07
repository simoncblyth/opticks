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

// om-;TEST=NGPUTest om-t 

#include "OPTICKS_LOG.hh"
#include "SStr.hh"
#include "NPY.hpp"
#include "NGPU.hpp"



void test_save()
{
    NGPU* gpu = NGPU::GetInstance() ; 

    gpu->add( 100, "name0", "owner0" ) ; 
    gpu->add( 200, "name1", "owner1" ) ; 
    gpu->add( 300, "name2345678", "owner2345678" ) ; 

    const char* path = "/tmp/NGPU.npy" ; 
    gpu->saveBuffer(path); 

/*
epsilon:npy blyth$ xxd /tmp/NGPU.npy 
00000000: 934e 554d 5059 0100 4600 7b27 6465 7363  .NUMPY..F.{'desc
00000010: 7227 3a20 273c 7538 272c 2027 666f 7274  r': '<u8', 'fort
00000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
00000030: 652c 2027 7368 6170 6527 3a20 2833 2c20  e, 'shape': (3, 
00000040: 3429 2c20 7d20 2020 2020 2020 2020 200a  4), }          .
00000050: 6e61 6d65 3000 0000 6f77 6e65 7230 0000  name0...owner0..
00000060: 0000 0000 0000 0000 6400 0000 0000 0000  ........d.......
00000070: 6e61 6d65 3100 0000 6f77 6e65 7231 0000  name1...owner1..
00000080: 0000 0000 0000 0000 c800 0000 0000 0000  ................
00000090: 6e61 6d65 3233 3435 6f77 6e65 7232 3334  name2345owner234
000000a0: 0000 0000 0000 0000 2c01 0000 0000 0000  ........,.......
epsilon:npy blyth$ 

*/
}

void test_load(const char* path)
{
    NGPU* gpu = NGPU::Load(path) ; 
    if(gpu) 
    {
        gpu->dump();  
    }
    else
    {
        LOG(info) << "no NGPU stats to load from: " << path ;      
    }
}


int main(int argc, char** argv )
{
    OPTICKS_LOG(argc, argv);

    const char* path = argc > 1 ? argv[1] : "$TMP/OKTest_NGPU.npy" ;  

    //test_save();
    test_load(path);

    return 0 ; 
}


