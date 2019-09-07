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


#include "NPY.hpp"
#include "MaterialLibNPY.hpp"
#include "PLOG.hh"

MaterialLibNPY::MaterialLibNPY(NPY<float>* mlib) 
       :  
       m_lib(mlib)
{
}


void MaterialLibNPY::dump(const char* msg)
{
    unsigned int ni = m_lib->m_ni ;
    unsigned int nj = m_lib->m_nj ;
    unsigned int nk = m_lib->m_nk ;

    LOG(info) << msg 
              << " ni " << ni 
              << " nj " << nj 
              << " nk " << nk 
              ; 

    assert( nj == 39 && nk == 4 );

    for(unsigned int i=0 ; i<ni ; i++ )
    {
        dumpMaterial(i);
    }
}

void MaterialLibNPY::dumpMaterial(unsigned int)
{
    

}




