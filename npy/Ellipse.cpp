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

#include "Ellipse.hpp"

const unsigned ellipse::NSTEP = 1000000u ; 

ellipse::ellipse( double ex, double ey )
{
    hemi.x = ex ; 
    hemi.y = ey ; 
}

/**
ellipse::closest_approach_to_point
------------------------------------

Returns a 2d coordinate of the point on the 
ellipse that is closest to the 2d point provided 
in the argument.

**/

glm::dvec2 ellipse::closest_approach_to_point( const glm::dvec2& p )
{
    const double pi = glm::pi<double>() ;
    double closest = std::numeric_limits<double>::max() ; 
    glm::dvec2 ret(0,0) ; 

    for(unsigned i=0 ; i < NSTEP ; i++ )
    {
         double t = double(i)*2.*pi/double(NSTEP) ; 
         glm::dvec2 e( hemi.x*cos(t) , hemi.y*sin(t) ); 
         double d = glm::distance2( e, p  ) ;  

         if( d < closest )
         {
              closest = d ; 
              ret = e ; 
         }  
    }
    return ret ; 
}

/*
ana/shape.py

def ellipse_closest_approach_to_point( ex, ez, _c ):
    """ 
    Ellipse natural frame, semi axes ex, ez.  _c coordinates of point

    :param ex: semi-major axis 
    :param ez: semi-major axis 
    :param c: xz coordinates of point 

    :return p: point on ellipse of closest approach to center of torus circle

    Closest approach on the bulb ellipse to the center of torus "circle" 
    is a good point to target for hype/cone/whatever neck, 
    as are aiming to eliminate the cylinder neck anyhow

    equation of RHS torus circle, in ellipse frame

        (x - R)^2 + (z - z0)^2 - r^2 = 0  

    equation of ellipse

        (x/ex)^2 + (z/ez)^2 - 1 = 0 

    """
    c = np.asarray( _c )   # center of RHS torus circle
    assert c.shape == (2,)

    t = np.linspace( 0, 2*np.pi, 1000000 )
    e = np.zeros( [len(t), 2] )
    e[:,0] = ex*np.cos(t) 
    e[:,1] = ez*np.sin(t)   # 1M parametric points on the ellipse 

    p = e[np.sum(np.square(e-c), 1).argmin()]   # point on ellipse closest to c 
    return p 

*/



