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

#include <sstream>
#include "PLOG.hh"

#include "NPlaneFromPoints.hpp"


std::string NPlaneFromPoints::desc() const 
{
    std::stringstream ss ; 
    ss 
       << " n " << points.size()
       << " cog " << glm::to_string(cog) 
       << " nrm " << glm::to_string(nrm) 
       ;

    return ss.str();
}

void NPlaneFromPoints::add(float x, float y, float z)
{
    glm::vec3 p(x,y,z);
    points.push_back(p);
}

void NPlaneFromPoints::add(const glm::vec3& p)
{  
    points.push_back(p);
}
void NPlaneFromPoints::update()
{
    update_cog();
    update_nrm();
    update_projection();
}

void NPlaneFromPoints::update_cog()
{
    cog.x = 0 ; 
    cog.y = 0 ; 
    cog.z = 0 ; 
    for(unsigned i=0 ; i < points.size() ; i++)
    { 
        glm::vec3 p = points[i] ; 
        cog += p ; 
    }
    cog /= points.size();
}

void NPlaneFromPoints::update_nrm()
{
    xx = 0 ; 
    xy = 0 ; 
    xz = 0 ; 
    yy = 0 ; 
    yz = 0 ; 
    zz = 0 ; 

    for(unsigned i=0 ; i < points.size() ; i++)
    { 
        glm::vec3 d = points[i] - cog ; 

        xx += d.x * d.x;
        xy += d.x * d.y;
        xz += d.x * d.z;
        yy += d.y * d.y;
        yz += d.y * d.z;
        zz += d.z * d.z;
    }
 
    det.x = yy*zz - yz*yz ;
    det.y = xx*zz - xz*xz ;
    det.z = xx*yy - xy*xy ;

    float det_max = std::max( std::max(det.x, det.y), det.z ) ; 
    assert( det_max > 0);

    float a, b ; 

    if(det_max == det.x)
    {
        a = (xz*yz - xy*zz) / det.x;
        b = (xy*yz - xz*yy) / det.x;
        nrm[0] = 1. ; 
        nrm[1] = a ; 
        nrm[2] = b ; 
    }
    else if (det_max == det.y) 
    {
        a = (yz*xz - xy*zz) / det.y;
        b = (xy*xz - yz*xx) / det.y;

        nrm[0] = a ; 
        nrm[1] = 1 ; 
        nrm[2] = b ; 
    } 
    else
    {
        a = (yz*xy - xz*yy) / det.z;
        b = (xz*xy - yz*xx) / det.z;
        nrm[0] = a ; 
        nrm[1] = b ; 
        nrm[2] = 1 ; 
    }
}



void NPlaneFromPoints::update_projection()
{  
    for(unsigned i=0 ; i < points.size(); i++)
    {
        glm::vec3 p = points[i]  ;  
        glm::vec3 proj ;  
        project_into_plane( proj, p );

/*
        std::cout << std::setw(4) << i 
                  << " p " << glm::to_string(p) 
                  << " proj " << glm::to_string(proj) 
                  << std::endl ; 

*/
        projection.push_back(proj);
    }
}

void NPlaneFromPoints::project_into_plane( glm::vec3& proj, const glm::vec3& p) const 
{
    glm::vec3 d = p - cog ;
    proj = d - glm::dot( d, nrm )*nrm ;  
}


float NPlaneFromPoints::azimuthal_diff( unsigned j ) const 
{
    assert( j < projection.size() );

    glm::vec3 pref = glm::normalize( projection[reference] ) ; 
    glm::vec3 proj = glm::normalize( projection[j] ); 

    return glm::dot(pref, proj) ;  
}

float NPlaneFromPoints::azimuthal_diff( const glm::vec3& p ) const 
{
    glm::vec3 pref = glm::normalize( projection[reference] ) ; 

    glm::vec3 proj ;
    project_into_plane( proj, p );

    glm::vec3 projn = glm::normalize(proj) ;

    return glm::dot(pref, projn) ;  
}


void NPlaneFromPoints::dump( const char* msg) const 
{
    LOG(info) << msg ; 
    assert( points.size() == projection.size() );

    for(unsigned i=0 ; i < points.size(); i++)
    {
        glm::vec3 p = points[i]  ;  
        glm::vec3 proj = projection[i] ;  
        float azd = azimuthal_diff(i);

        std::cout << std::setw(4) << i 
                  << " p " << glm::to_string(p) 
                  << " proj " << glm::to_string(proj) 
                  << " azd " << azd
                  << std::endl ; 

    }

}







