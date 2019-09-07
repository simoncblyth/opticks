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
#include <algorithm>

#include "NGLM.hpp"

#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"

#include "AxisNPY.hpp"
#include "PLOG.hh"


AxisNPY::AxisNPY(NPY<float>* axis) 
       :  
       m_axis(axis)
{
}
NPY<float>* AxisNPY::getAxis()
{
    return m_axis ; 
}

void AxisNPY::dump(const char* msg)
{
    if(!m_axis) return ;
    LOG(info) << msg ; 

    unsigned int ni = m_axis->m_ni ;
    unsigned int nj = m_axis->m_nj ;
    unsigned int nk = m_axis->m_nk ;
    assert( nj == 3 && nk == 4 );


    for(unsigned int i=0 ; i < ni ; i++)
    {

        glm::vec4 vpos = m_axis->getQuad(i,0);
        glm::vec4 vdir = m_axis->getQuad(i,1);
        glm::vec4 vcol = m_axis->getQuad(i,2);

        printf("%2u %s %s %s  \n", 
                i, 
                gpresent(vpos,2,11).c_str(),
                gpresent(vdir,2,7).c_str(),
                gpresent(vcol,2,7).c_str()
             );
    }
}


