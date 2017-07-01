#include <sstream>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "GLMFormat.hpp"
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include <glm/gtx/component_wise.hpp>
#include "NBBox.hpp"


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
    tbb.side = tbb.max - tbb.min ;
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
    tbb.side = tbb.max - tbb.min ;
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

bool nbbox::FindOverlap(nbbox& overlap, const nbbox& a, const nbbox& b)
{
    if(!HasOverlap(a,b)) 
    {
        overlap.empty = true ; 
        return false ; 
    }

    overlap.min.x = fmaxf(a.min.x, b.min.x) ;
    overlap.min.y = fmaxf(a.min.y, b.min.y) ;
    overlap.min.z = fmaxf(a.min.z, b.min.z) ;

    overlap.max.x = fminf(a.max.x, b.max.x) ;
    overlap.max.y = fminf(a.max.y, b.max.y) ;
    overlap.max.z = fminf(a.max.z, b.max.z) ;

    overlap.side = overlap.max - overlap.min ; 

    return true ; 
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
    // see csgbbox- for searches for papers to help with an algebra of CSG bbox 
    std::string expr ; 

    comb.invert = false ; // set true for unbounders
    comb.empty = false ;  // set true when no overlap found

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
            expr = " BB(A - B)  -> BB(A)  " ;
            comb.include(a); 
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
    ss
        << " mi " << min.desc() 
        << " mx " << max.desc() 
        << " si " << side.desc() 
        << ( invert ? " INVERTED" : "" )
        << ( empty ? " EMPTY" : "" )
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
    min.dump("bb min");
    max.dump("bb max");
}

void nbbox::include(const nbbox& other)
{
    min = nminf( min, other.min );
    max = nmaxf( max, other.max );
    side = max - min ; 
}


void nbbox::include(const glm::vec3& p)
{
    nvec3 pp = {p.x, p.y, p.z } ;

    min = nminf( min, pp );
    max = nmaxf( max, pp );
    side = max - min ; 
}

nbbox nbbox::from_points(const std::vector<glm::vec3>& points)
{
    nbbox bb ;
    bb.empty = false ; 
    bb.invert = false ; 

    for(unsigned i=0 ; i < points.size() ; i++) bb.include(points[i]) ;
    return bb ; 
}




bool nbbox::contains(const nvec3& p, float epsilon) const 
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






