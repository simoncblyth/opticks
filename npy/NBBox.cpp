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
#include <iostream>
#include <iomanip>
#include <cstring>

#include "GLMFormat.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "nmat4triple.hpp"
#include <glm/gtx/component_wise.hpp>
#include "NBBox.hpp"
#include "NPoint.hpp"

#include "PLOG.hh"


const plog::Severity nbbox::LEVEL = PLOG::EnvLevel("NBBox","DEBUG"); 

nbbox nbbox::make_transformed( const glm::mat4& t ) const 
{
    nbbox tbb(*this) ; // <-- default copy ctor copies over "invert" and "empty"
    transform( tbb, *this, t );
    return tbb ; 
}

void nbbox::transform_brute(nbbox& tbb, const nbbox& bb, const glm::mat4& t )
{
    glm::vec4 min(bb.min.x, bb.min.y, bb.min.z , 1.f);
    glm::vec4 max(bb.max.x, bb.max.y, bb.max.z , 1.f);
    glm::vec4 dif(max - min);

    glm::vec4 t_min(FLT_MAX) ; 
    glm::vec4 t_max(-FLT_MAX) ; 

    for(int i=0 ; i < 8 ; i++) // over corners in Morton z-order
    {
        glm::vec4 corner(min.x + ( i & 1 ? dif.x : 0 ), min.y + ( i & 2 ? dif.y : 0 ), min.z + ( i & 4 ? dif.z : 0 ), 1.f ); 
        glm::vec4 t_corner = t * corner ; 
        gminf( t_min, t_min, t_corner );
        gmaxf( t_max, t_max, t_corner );
    }

    tbb.min = { t_min.x , t_min.y, t_min.z } ;
    tbb.max = { t_max.x , t_max.y, t_max.z } ;
}

void nbbox::transform(nbbox& tbb, const nbbox& bb, const glm::mat4& t )
{
    // http://dev.theomader.com/transform-bounding-boxes/

    glm::vec4 xa = t[0] * bb.min.x ; 
    glm::vec4 xb = t[0] * bb.max.x ; 
    glm::vec4 xmi = gminf(xa, xb );      
    glm::vec4 xma = gmaxf(xa, xb );      

    glm::vec4 ya = t[1] * bb.min.y ; 
    glm::vec4 yb = t[1] * bb.max.y ; 
    glm::vec4 ymi = gminf(ya, yb );      
    glm::vec4 yma = gmaxf(ya, yb );      

    glm::vec4 za = t[2] * bb.min.z ; 
    glm::vec4 zb = t[2] * bb.max.z ; 
    glm::vec4 zmi = gminf(za, zb );      
    glm::vec4 zma = gmaxf(za, zb );      
     
    glm::vec4 t_min = xmi + ymi + zmi + t[3] ; 
    glm::vec4 t_max = xma + yma + zma + t[3] ; 

    tbb.min = { t_min.x , t_min.y, t_min.z } ;
    tbb.max = { t_max.x , t_max.y, t_max.z } ;
}

std::string nbbox::containment_mask_string( unsigned mask )
{
    return nbboxenum::ContainmentMaskString(mask);
}
const char* nbbox::containment_name( NBBoxContainment_t cont )
{
    return nbboxenum::ContainmentName(cont);
}

NBBoxContainment_t nbbox::classify_containment_1( float delta, float epsilon,  NBBoxContainment_t neg, NBBoxContainment_t eps, NBBoxContainment_t pos )  // static
{
    /*

               eps
           -ve  ||   +ve
                || 


    The bbox must be transformed to be in the same frame.


                    .      container.max
          +-------------------+
          |                   |  
          |        this.max   |
          |   +----------+    |
          |   |     .    |    | 
          |   |          |    |
          |   +----------+    |
          |  this.min         |
          +-------------------+
     container.min

    */
    NBBoxContainment_t cont = UNCLASSIFIED ; 
    if( fabsf(delta) < epsilon ) cont = eps ;  
    else if( delta < 0.f )       cont = neg ;  
    else if( delta > 0.f )       cont = pos ;  
    else                         assert(0); 

/*
    std::cout 
        << " nbbox::classify_containment_1" 
        << " delta " << std::setw(15) << std::fixed << std::setprecision(5) << delta 
        << " fabsf(delta) " << std::setw(15) << std::fixed << std::setprecision(5) << fabsf(delta) 
        << " epsilon " << std::setw(15) << std::fixed << std::setprecision(5) << epsilon
        << " cont " << containment_name(cont)
        << std::endl 
        ; 
*/

    return cont ; 
}

unsigned nbbox::classify_containment( const nbbox& container, float epsilon ) const  // of this bbox against purported container
{
    // hmm rotational transforms mess with bbox dimensions, except in axis direction

    glm::vec3 dmin( this->min.x - container.min.x, this->min.y - container.min.y, this->min.z - container.min.z );
    glm::vec3 dmax( this->max.x - container.max.x, this->max.y - container.max.y, this->max.z - container.max.z );

    unsigned mask = 0 ;  

    mask |= classify_containment_1( dmin.x , epsilon, XMIN_OUTSIDE, XMIN_COINCIDENT, XMIN_INSIDE ) ;
    mask |= classify_containment_1( dmin.y , epsilon, YMIN_OUTSIDE, YMIN_COINCIDENT, YMIN_INSIDE ) ;
    mask |= classify_containment_1( dmin.z , epsilon, ZMIN_OUTSIDE, ZMIN_COINCIDENT, ZMIN_INSIDE ) ;

    mask |= classify_containment_1( dmax.x , epsilon, XMAX_INSIDE, XMAX_COINCIDENT, XMAX_OUTSIDE ) ;
    mask |= classify_containment_1( dmax.y , epsilon, YMAX_INSIDE, YMAX_COINCIDENT, YMAX_OUTSIDE ) ;
    mask |= classify_containment_1( dmax.z , epsilon, ZMAX_INSIDE, ZMAX_COINCIDENT, ZMAX_OUTSIDE ) ;

    unsigned expected = XMIN_INSIDE | YMIN_INSIDE | ZMIN_INSIDE | XMAX_INSIDE | YMAX_INSIDE | ZMAX_INSIDE ;
    unsigned errmask = mask & ~expected  ;  // clear expected bits 

/*
    std::cout 
        << " nbbox::classify_containment" 
        << " this " << desc()  << std::endl 
        << " cont " << container.desc() << std::endl
        << " dmin (this-container) " << glm::to_string(dmin) << std::endl
        << " dmax (this-container) " << glm::to_string(dmax) << std::endl 
        << " mask    " << containment_mask_string(mask) << std::endl 
        << " errmask " << containment_mask_string(errmask) << std::endl 
        ;
*/

    return errmask ; 
}




bool nbbox::HasOverlap(const nbbox& a, const nbbox& b )
{
    if( a.max.x < b.min.x || a.min.x > b.max.x ) return false ; 
    if( a.max.y < b.min.y || a.min.y > b.max.y ) return false ; 
    if( a.max.z < b.min.z || a.min.z > b.max.z ) return false ; 
    return true ; 
}

float nbbox::diagonal() const 
{
    return glm::length(max-min);
}

bool nbbox::FindOverlap(nbbox& overlap, const nbbox& a, const nbbox& b)
{
    if(!HasOverlap(a,b)) 
    {
        overlap.set_empty() ; 
        return false ; 
    }

    overlap.min.x = fmaxf(a.min.x, b.min.x) ;
    overlap.min.y = fmaxf(a.min.y, b.min.y) ;
    overlap.min.z = fmaxf(a.min.z, b.min.z) ;

    overlap.max.x = fminf(a.max.x, b.max.x) ;
    overlap.max.y = fminf(a.max.y, b.max.y) ;
    overlap.max.z = fminf(a.max.z, b.max.z) ;

    return true ; 
}


bool nbbox::inside_range(const float v, const float vmin, const float vmax ) 
{
    return v > vmin && v < vmax ; 
}

void nbbox::SubtractOverlap(nbbox& result, const nbbox& a, const nbbox& o, int verbosity )
{
    /*
    Overlap *o* must be the overlap of *a* with another box, this simplifies 
    the conditions to handle.
    
    * small contained overlap cutaways (eg chopping off a corner or a hole in middle) 
      cannot be expressed as a bbox so they dont change the bbox : just leave at *a* 

    * overlaps need to be complete along two dimensions in order to 
      be able to chop along the other axis 


    * if there is a chop axis (say z), then just need to know whether
      the need to change min.z or max.z to effect the chop


       +-----------+
       |  a        |
       *- - - - - -*
       |           |
       |  o        |
       +-----------+

    */

    result = a ; 
    if(o.is_empty()) return ; 

    for(unsigned i=0 ; i < 3 ; i++) // test each axis for potential chop, ie when there is complete coverage in other two directions
    {
        unsigned j = (i + 1) % 3 ;  
        unsigned k = (i + 2) % 3 ; 

        bool jk_match = 
                         o.min[j] == a.min[j]  &&
                         o.min[k] == a.min[k]  &&
                         o.max[j] == a.max[j]  &&
                         o.max[k] == a.max[k]   ;


        bool omax_i_inside = inside_range(o.max[i], a.min[i], a.max[i]) ;
        bool omin_i_inside = inside_range(o.min[i], a.min[i], a.max[i]) ;


        if(verbosity > 2)
        std::cout 
                 << " i " << std::setw(1) << i
                 << " j " << std::setw(1) << j 
                 << " k " << std::setw(1) << k 
                 << " jk_match " << ( jk_match ? "Y" : "N" )
                 << " omax_i_inside " << ( omax_i_inside ? "Y" : "N" )
                 << " omin_i_inside " << ( omin_i_inside ? "Y" : "N" )
                 << " a.min[i] " << std::setw(10) << std::fixed << std::setprecision(3) << a.min[i]
                 << " a.max[i] " << std::setw(10) << std::fixed << std::setprecision(3) << a.max[i]
                 << " o.min[i] " << std::setw(10) << std::fixed << std::setprecision(3) << o.min[i]
                 << " o.max[i] " << std::setw(10) << std::fixed << std::setprecision(3) << o.max[i]
                 << std::endl ; 

 
        if(omax_i_inside)
        {
            if(verbosity > 2)
            std::cout << "pulling up a.min ie chopping off below " << std::endl ;

            result.min[i] = o.max[i] ;    
        }
        else if(omin_i_inside)
        {
            if(verbosity > 2)
            std::cout << "pulling down a.max ie chopping off above " << std::endl ;

            result.max[i] = o.min[i] ;    
        }
    }
}





bool nbbox::has_overlap(const nbbox& other)
{
    return HasOverlap(*this, other);
}
bool nbbox::find_overlap(nbbox& overlap, const nbbox& other)
{
    return FindOverlap(overlap, *this, other);
}

void nbbox::CombineCSG(nbbox& comb, const nbbox& a, const nbbox& b, OpticksCSG_t op, int verbosity )
{
/*

Obtaining the BBOX of a CSG tree is non-trivial
===================================================

Alternative Approach
----------------------

* perhaps these complications can be avoiding by forming a bbox
  from the composite parametric points (ie look at all parametric 
  points of all primitives transformed into CSG tree root frame and 
  make a selection based on their composite SDF values... points
  within epsilon of zero are regarded as being on the composite 
  surface). 

  As the parametric points should start exactly at SDF zero 
  for the primitives, and they are transformed only rather locally 
  I expect that a very tight epsilon 1e-5 should be appropriate.


Analytic BB(CSG) approach
---------------------------

* see csgbbox- for searches for papers to help with an algebra of CSG bbox 
  and a look at how OpenSCAD and povray handle this  

* best paper found on this by far is summarised below


Computing CSG tree boundaries as algebraic expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Marco Mazzetti  
Luigi Ciminiera 

* http://dl.acm.org/citation.cfm?id=164360.164416
* ~/opticks_refs/csg_tree_boundaries_as_expressions_p155-mazzetti.pdf

Summary of the paper:

* bbox obtained from a CSG tree depends on evaluation order !!, 
  as the bbox operation is not associative, 

* solution is to rearrange the boolean expression tree into 
  a canonical form (UOI : union-of-intersections, aka sum-of-products) 
  which the paper states corresponds to the minimum bbox


* upshot of this is that generally the bbox obtained will be overlarge

* handling CSG difference requires defining an InnerBB 
  corresponding to the maximum aabb that is completely inside the shape, 
  then::

      BB(A - B) = BB(A) - InnerBB(B)


*/

    std::string expr ; 

    comb.invert = false ; // set true for unbounders

    if(op == CSG_INTERSECTION )
    {
        if(!a.invert && !b.invert)     
        {
            expr = " BB(A * B) " ; 
            FindOverlap(comb, a, b);
        }
        else if(!a.invert && b.invert)  
        {
            expr = " BB( A * !B  ->  A - B ) ->  BB(A) " ; 
            comb.include(a);   
        }
        else if(a.invert && !b.invert)
        {
            expr = " BB(!A * B  ->  B - A ) ->  BB(B)  " ; 
            comb.include(b);   
        }
        else if(a.invert && b.invert) 
        {
            expr = " BB(!A * !B  ->  !(A+B) ) ->  !BB(A+B)   " ; 
            comb.include(a);
            comb.include(b);
            comb.invert = true ;
        }
    } 
    else if (op == CSG_UNION )
    {
        if(!a.invert && !b.invert)    
        {
            expr = "  BB(A+B) " ; 
            comb.include(a);
            comb.include(b);
        } 
        else if(a.invert && !b.invert) 
        {
            expr = " BB( !A+B ->  !( A*!B ) -> !( A-B ) ) ->   ! BB(A)  " ; 
            comb.include(a);
            comb.invert = true ;
        } 
        else if(!a.invert && b.invert)  
        {
            expr = " BB( A+!B -> !( !A*B ) -> !( B-A ) ) ->   ! BB(B)   " ;
            comb.include(b);
            comb.invert = true ;
        } 
        else if(a.invert && b.invert)   
        {
            expr = "  BB( !A+!B ->  !( A*B ) ) ->   ! BB(A*B)   " ; 
            FindOverlap(comb, a, b);
            comb.invert = true ; 
        }
    }
    else if( op == CSG_DIFFERENCE )
    {
        if(!a.invert && !b.invert)  
        {
            expr = " BB(A - B)  -> BB(A)  " ;  // hmm can do better than this by considering A - B ->  A*!B
            comb.include(a); 

            bool experimental_bbox_subtraction = false ; // needs more dev ... fixes some but makes many worse, 
            if(experimental_bbox_subtraction)
            {
                nbbox a_overlap ; 
                FindOverlap(a_overlap, a, b );
                SubtractOverlap(comb, a, a_overlap, verbosity);
            } 
        }
        else if( a.invert && b.invert)   
        {
            expr = " BB(  !A - !B -> !A * B -> B*!A ->  B - A ) -> BB(B) " ;  
            comb.include(b);
        }
        else if( !a.invert && b.invert)  
        {
            expr = " BB( A - !B  -> A * B ) -> BB(A*B) " ; 
            FindOverlap(comb, a, b);
        }
        else if( a.invert && !b.invert)  
        {
            expr = " BB( !A - B -> !A * !B -> !( A+B ) ) -> !BB(A+B)    " ; 
            comb.include(a);
            comb.include(b);
            comb.invert = true ; 
        }
    }



    if(verbosity > 0)
    std::cout 
               << "nbbox::CombineCSG"
               << " " << expr << std::endl 
               << " L " << a.desc()    << std::endl 
               << " R " << b.desc()    << std::endl 
               << " C " << comb.desc() << std::endl
               ;

}


/*

BB(A - B)
=============

How to calculate the box min/max from the
subtraction of two boxes ?  ...

* 1st calulate the overlap A*B 

::

         +----------+     
         |          |
         |          |
         |          |
         |          |
         | A        | 
         |          |
     +---*----------*---+
     |   |  A*B     |   |
     |   +----------+   |
     | B                |
     +------------------+



    A - B = A*!B

    A - (A*B) 


Comparing A with A*B i can enumerate the cases...
 



With the overlap 

         +----------+
         |          |             
         |          |             
         |  A - B   |             
         |          |             
         |          |             
         |          |
         *----------*     

         *----------*     
         |   A*B    |
         +----------+


*/





std::function<float(float,float,float)> nbbox::sdf() const 
{
    return *this ;     
}

float nbbox::operator()(float x_, float y_, float z_, const nmat4triple* t_ ) const 
{
    glm::vec3 tmp(x_, y_, z_);
    return sdf_(tmp, t_ );
}
float nbbox::operator()(const glm::vec3& q, const nmat4triple* t_ ) const 
{
    return sdf_(q, t_);
}
float nbbox::sdf_(const glm::vec3& q_, const nmat4triple* t_ ) const 
{
    glm::vec4 p(q_, 1.0);  

    if(t_) p = t_->v * p ;  // apply inverse transform on query point 
 

    glm::vec3 bmi(min.x, min.y, min.z);
    glm::vec3 bmx(max.x, max.y, max.z);
    glm::vec3 bce = (bmi + bmx)/2.f ; 
    glm::vec3 bhs = (bmx - bmi)/2.f ; 

    glm::vec3 q = glm::vec3(p) - bce ; 

    // below works for a symmetric box at origin 
    // ... in which case bmx is really the halfside
    // so the above applies an inverse translation to the query point to
    // dispose the box into that position 
    // see NBBoxTest.cc:test_sdf

    glm::vec3 d = glm::abs(q) - bhs  ;

    float dmaxcomp = glm::compMax(d);

    glm::vec3 dmax = glm::max( d, glm::vec3(0.f) );

    float d_inside = fminf(dmaxcomp, 0.f);
    float d_outside = glm::length( dmax );

    return d_inside + d_outside ;       

}


void nbbox::scan_sdf( const glm::vec3& o, const glm::vec3& range, const nmat4triple* t ) const 
{
    const nbbox& bb = *this ; 
    std::cout 
         << "nbbox::scan_sdf"
         << "bb" << bb.desc()  
         << std::endl 
         << gpresent("ori", o) 
         << gpresent("range", range) 
         << ( t ? gpresent("t.t", t->t ) : "" )
         << std::endl  
         ;

    for(float w=range.x ; w < range.y ; w+=range.z )
        std::cout 
                  << "         w " << std::setw(10) << std::fixed << std::setprecision(3) << w
                  << " sd(w,w,w) " << std::setw(10) << std::fixed << std::setprecision(3) << bb(o.x+w,o.y+w,o.z+w,t)  
                  << " sd(w,0,0) " << std::setw(10) << std::fixed << std::setprecision(3) << bb(o.x+w,o.y+0,o.z+0,t)  
                  << " sd(0,w,0) " << std::setw(10) << std::fixed << std::setprecision(3) << bb(o.x+0,o.y+w,o.z+0,t)  
                  << " sd(0,0,w) " << std::setw(10) << std::fixed << std::setprecision(3) << bb(o.x+0,o.y+0,o.z+w,t)  
                  << std::endl 
                  ;
}


std::string nbbox::description() const 
{
    std::stringstream ss ; 
    glm::vec3 si = side();

    ss
        << " mi " << gpresent(min)
        << " mx " << gpresent(max) 
        << " si " << gpresent(si) 
        << ( invert ? " INVERTED" : "" )
        << ( is_empty() ? " EMPTY" : "" )
        ;

    return ss.str();
}

const char* nbbox::desc() const
{
    //char _desc[128];
    //snprintf(_desc, 128, " mi %.32s mx %.32s ", min.desc(), max.desc() );
    std::string _desc = description() ; 
    return strdup(_desc.c_str());
}


void nbbox::dump(const char* msg)
{
    printf("%s\n", msg);
    std::cout 
       << " bb.mi " << gpresent(min) 
       << " bb.mx " << gpresent(max)
       << std::endl ;
 
}

void nbbox::include(const nbbox& other)
{
    if(is_empty())
    {
        min = other.min ; 
        max = other.max ;
    }
    else
    { 
        //min = nminf( min, other.min );
        //max = nmaxf( max, other.max );
        min = glm::min( min, other.min );
        max = glm::max( max, other.max );


    }
}

void nbbox::include(const glm::vec4& p_)
{
    glm::vec3 p(p_); 
    include(p); 
}

void nbbox::include(const glm::vec3& p)
{
    if(is_empty())
    {
        min = p ; 
        max = p ; 
    }
    else
    {
        min = glm::min( min, p );
        max = glm::max( max, p );
    }
}

/*

      +-  - - - -*   <--- included point pushing out the max, leaves min unchanged
      .          | 
      +-------+  .
      |       |  |
      |       |  .
      |       |  |
      +-------+- +

      +-------+  
      |    *  |     <-- interior point doesnt change min/max  
      |       |  
      |       |  
      +-------+ 

      +-------+-->--+  
      |       |     |  
      |       |     *  <--- side point pushes out max, leaves min unchanged
      |       |     |
      +-------+-----+ 

*/



nbbox nbbox::from_points(const std::vector<glm::vec3>& points, unsigned verbosity)
{
    nbbox bb = make_bbox() ;
    assert( bb.is_empty() );
    assert( bb.invert == false );

    LOG(LEVEL)
       << " num_points " << points.size()
       << " bb0 " << bb.desc()
       ;


    for(unsigned i=0 ; i < points.size() ; i++) 
    {
        glm::vec3 p = points[i]; 

        bb.include(p) ;

        if(verbosity > 5)
            std::cout 
                 << " i " << std::setw(4) << i 
                 << " p " << gpresent(p)
                 << " bb " << bb.desc()
                 << std::endl
                 ;

    }
    return bb ; 
}

nbbox nbbox::from_points(const NPoint* points)
{
    nbbox bb = make_bbox() ;
    assert( bb.is_empty() );
    assert( bb.invert == false );

    unsigned n = points->getNum(); 
    for(unsigned i=0 ; i < n ; i++)
    {
        glm::vec4 p = points->get(i); 
        bb.include(p); 
    }
    return bb ; 
}






bool nbbox::contains(const nvec3& p, float epsilon) const 
{
   glm::vec3 pp(p.x, p.y, p.z);
   return contains(pp, epsilon);
}

bool nbbox::contains(const glm::vec3& p, float epsilon) const 
{
    // allow epsilon excursions
    bool x_ok = p.x - min.x > -epsilon && p.x - max.x < epsilon ;
    bool y_ok = p.y - min.y > -epsilon && p.y - max.y < epsilon ;
    bool z_ok = p.z - min.z > -epsilon && p.z - max.z < epsilon ;

    bool xyz_ok = x_ok && y_ok && z_ok ; 

/*
    std::cout << "nbbox::contains"
               << " epsilon " << epsilon 
               << " x_ok " << x_ok
               << " y_ok " << y_ok
               << " z_ok " << z_ok
               << " xyz_ok " << xyz_ok
               << std::endl ; 

    if(!x_ok) std::cout 
              << "nbbox::contains(X)"
              << " epsilon " << epsilon 
              << " p.x " << p.x
              << " min.x " << min.x 
              << " max.x " << max.x 
              << std::endl ;
 
    if(!y_ok) std::cout
              << "nbbox::contains(Y)"
              << " epsilon " << epsilon 
              << " p.y " << p.y
              << " min.y " << min.y 
              << " max.y " << max.y 
              << std::endl ;
 
    if(!z_ok) std::cout 
              << "nbbox::contains(Z)"
              << " epsilon " << epsilon 
              << " p.z " << p.z
              << " min.z " << min.z 
              << " max.z " << max.z 
              << std::endl ;
 */


    return xyz_ok ; 

} 

bool nbbox::contains(const nbbox& other, float epsilon ) const
{
    return contains( other.min, epsilon ) && contains(other.max, epsilon ) ;
} 


float nbbox::MaxDiff( const nbbox& a, const nbbox& b)
{
    glm::vec3 dmn = glm::abs(a.min - b.min) ;
    glm::vec3 dmx = glm::abs(a.max - b.max) ;

    return std::max<float>( glm::compMax(dmn), glm::compMax(dmx) );
}



void nbbox::copy_from(const nbbox& src)
{
    min = src.min ; 
    max = src.max ; 
    invert = src.invert ; 
}


float nbbox::extent(const nvec4& dim) 
{
    float _extent(0.f) ;
    _extent = nmaxf( dim.x , _extent );
    _extent = nmaxf( dim.y , _extent );
    _extent = nmaxf( dim.z , _extent );
    _extent = _extent / 2.0f ;    
    return _extent ; 
}

nvec4 nbbox::dimension_extent() const
{
    nvec4 de ; 
    de.x = max.x - min.x ; 
    de.y = max.y - min.y ; 
    de.z = max.z - min.z ; 
    de.w = extent(de) ; 
    return de ; 
}

nvec4 nbbox::center_extent() const 
{
    nvec4 ce ; 
    ce.x = (min.x + max.x)/2.f ;
    ce.y = (min.y + max.y)/2.f ;
    ce.z = (min.z + max.z)/2.f ;
    nvec4 de = dimension_extent();
    ce.w = de.w ;  
    return ce ; 
}

glm::vec4 nbbox::ce() const 
{
    nvec4 v = center_extent() ; 
    return glm::vec4(v.x, v.y, v.z, v.w )  ; 
}




