#include <limits>
#include <set>

#include "NPY.hpp"
#include "NGLM.hpp"
#include "IntersectSDF.hh"
#include "PLOG.hh"

#include "NBox.hpp"
#include "NSphere.hpp"

/**
IntersectSDF
==============

This does in C++ essentially the same as the python
examples/UseOptiXGeometryInstancedOCtx/intersect_sdf_test.py 

TODO:

* find way to generalize this to typical Opticks CSG geometries

**/


float IntersectSDF::sdf(unsigned geocode, const glm::vec3& lpos )
{
    float sz = 5.f ; 
    float radius = sz ; 
    glm::vec3 box(sz/2.f,sz/2.f,sz/2.f);  

    float sd = 0.f ; 
    switch(geocode)
    {
       case 1: sd =    nbox::sdf_local_(lpos, box)     ; break ; 
       case 2: sd = nsphere::sdf_local_(lpos, radius ) ; break ; 
    }
    return sd ; 
}

const plog::Severity IntersectSDF::LEVEL = PLOG::EnvLevel("IntersectSDF", "DEBUG"); 


unsigned IntersectSDF::getRC() const { return m_rc ; }

/**
1. NB: transform identity integer is stuffed into spare [0,3] (top-right) slot in the 4x4 transform.
2. FixColumnFour sets it to [0,0,0,1] clearing the identity 
**/

IntersectSDF::IntersectSDF(const char* dir, float epsilon)
    :
    m_dir( strdup(dir) ),
    m_epsilon( epsilon ), 
    m_pixels(     NPY<unsigned char>::load(m_dir, "pixels.npy" ) ),
    m_posi(       NPY<float>::load(m_dir, "posi.npy" ) ),
    m_transforms( NPY<float>::load(m_dir, "transforms.npy" ) ),
    m_identity(   m_transforms ? ExtractTransformIdentity(m_transforms) : NULL),
    m_fixcount(   m_transforms ? FixColumnFour(m_transforms) : 0 ),
    m_itransforms( m_transforms ? NPY<float>::make_inverted_transforms(m_transforms) : NULL ), 
    m_rc( m_posi && m_transforms && m_identity && m_itransforms ? 0 : 1 )
{
    if( m_rc > 0 )
    {
        LOG(fatal) << " failed to load arrays from dir " << m_dir ;   
        return ; 
    }
    check_lpos_sdf();
}

std::string IntersectSDF::desc() const 
{
    std::stringstream ss ; 
    ss << " IntersectSDF "
       << " dir " << m_dir 
       << " pixels " << ( m_pixels ? m_pixels->getShapeString() : "-" ) 
       << " posi " << ( m_posi ? m_posi->getShapeString() : "-" )
       << " transforms " << ( m_transforms ? m_transforms->getShapeString() : "-" )
       << " identity " << ( m_identity ? m_identity->getShapeString() : "-" )
       << " fixcount " << m_fixcount 
       << " itransforms " << ( m_itransforms ? m_itransforms->getShapeString() : "-" )
       ;
    return ss.str(); 
}

NPY<unsigned>* IntersectSDF::ExtractTransformIdentity( const NPY<float>* transforms) // static 
{
    assert(transforms->hasShape(-1,4,4)); 
    return ExtractUInt(transforms, 0,3 ); 
}

NPY<unsigned>* IntersectSDF::ExtractUInt( const NPY<float>* src, unsigned j, unsigned k ) // static
{
    unsigned num_items = src->getNumItems(); 
    NPY<unsigned>* extract = NPY<unsigned>::make(num_items); 
    extract->zero(); 
    for(unsigned i=0 ; i < num_items ; i++)
    {
        unsigned value = src->getUInt(i,j,k,0) ; 
        extract->setValue(i,0,0,0, value); 
    }
    return extract ; 
}

unsigned IntersectSDF::FixColumnFour( NPY<float>* a ) // static
{
    assert( a->hasShape(-1,4,4) ); 
    unsigned ni = a->getShape(0); 
    unsigned nj = a->getShape(1);
    unsigned nk = a->getShape(2);
    unsigned l = 0 ; 

    unsigned count(0); 
    for(unsigned i=0 ; i < ni ; i++){
    for(unsigned j=0 ; j < nj ; j++){
    for(unsigned k=0 ; k < nk ; k++)
    {
        if( k == 3 ) 
        {
            float value = j == 3 ? 1.f : 0.f ;  
            a->setValue(i,j,k,l, value ); 
            count++ ; 
        }
    }
    }
    }
    return count ; 
}


/**
IntersectSDF::check_lpos_sdf
---------------------------------------------

1. for each shape select all unique transform indices.
2. for each transform index get the local frame intersect coordinates
3. for each local frame intersect coordinate compute the sdf giving 
   the distance to the surface, which is expected to be within epsilon of zero
4. record the min/max values of the sdf distance for each transform and geocode

**/


void IntersectSDF::check_lpos_sdf()
{
    typedef std::set<unsigned>::const_iterator IT ; 
    for(unsigned geocode=1 ;  geocode < 3 ; geocode++)
    {
        std::set<unsigned> tpx ;
        select_intersect_tranforms(tpx, geocode); 
        glm::vec2 g_mimx(std::numeric_limits<float>::max(), std::numeric_limits<float>::min() ) ; 
        unsigned lpos_tot(0); 

        for(IT it=tpx.begin() ; it != tpx.end() ; it++)
        {
            unsigned transform_index = *it ; 
            std::vector<glm::vec4> lpos ; 
            get_local_intersects(lpos, transform_index);
            glm::vec2 t_mimx( std::numeric_limits<float>::max(), std::numeric_limits<float>::min() ) ; 
            lpos_tot += lpos.size() ;  

            for(unsigned i=0 ; i < lpos.size() ; i++)
            {
                 const glm::vec4& lpo = lpos[i]; 
                 float sd = sdf(geocode, lpo) ; 

                 bool expect = std::abs(sd) < m_epsilon ; 
                 if(!expect)
                     LOG(fatal)
                         << " i " << std::setw(5) << i  
                         << " lpo " << glm::to_string( lpo )
                         << " sd " << sd
                         << " epsilon " << m_epsilon
                         << std::endl
                         ;
                 assert(expect); 

                 if( sd < t_mimx.x ) t_mimx.x = sd ; 
                 if( sd > t_mimx.y ) t_mimx.y = sd ; 

            }
            LOG(LEVEL) 
                << " geocode " << geocode  
                << " transform_index " << transform_index 
                << " lpos " << lpos.size() 
                << " t_mimx " << glm::to_string(t_mimx)
                ;  

            if( t_mimx.x < g_mimx.x ) g_mimx.x = t_mimx.x ; 
            if( t_mimx.y > g_mimx.y ) g_mimx.y = t_mimx.y ; 
        }

        LOG(info) 
            << " geocode " << geocode  
            << " tpx " << tpx.size()
            << " lpos_tot " << lpos_tot
            << " g_mimx " << glm::to_string(g_mimx)
            ;  
    }
}

/**
IntersectSDF::select_intersect_tranforms
-----------------------------------------

Loops over all pixels collecting all unique transform indices *tpx*
with the argument geocode, ie with box or sphere shapes.

**/

void IntersectSDF::select_intersect_tranforms(std::set<unsigned>& tpx, unsigned geocode)
{
    assert( m_posi->getDimensions() == 3 ); 
    unsigned ni = m_posi->getShape(0);  
    unsigned nj = m_posi->getShape(1);
    unsigned nk = m_posi->getShape(2);
    assert( nk == 4 ); 
    unsigned count(0);  

    for(unsigned i=0 ; i < ni ; i++){
    for(unsigned j=0 ; j < nj ; j++){

        unsigned pxid = m_posi->getUInt(i,j,3,0) ; 
        unsigned gc = pxid >> 24 ;       // geocode
        unsigned ti = pxid & 0xffffff ;  // transforms index
        bool select = gc == geocode  ;  
        if(!select) continue ;   
        
        count++ ;
        tpx.insert(ti); 

/*
        if(count < 100)
        std::cout
            << " count " 
            << std::setw(8) << count 
            << " (" 
            << std::setw(5) << i  
            << " " 
            << std::setw(5) << j
            << ")"
            << " gc " << gc
            << " ti " << std::setw(4) << ti 
            << std::endl 
            ;
*/
    }
    }

    LOG(debug) 
         << " posi " << m_posi->getShapeString()
         << " geocode " << geocode
         << " count " << count 
         << " tpx " << tpx.size()
         ; 
}

/**
IntersectSDF::get_local_intersects
-----------------------------------

Return in lpos the local 3d coordinates of pixels with the 
provided transform_index corresponding to all pixels for a particular instance.

*posi* array contains 3d position (global frame) and identity for every pixel 
in a pixel raster. This selects pixels with the argument transform_index
and transforms teh global positions for those pixels into local frame.

See examples/UseOptiXGeometryInstancedOCtx

**/

void IntersectSDF::get_local_intersects(std::vector<glm::vec4>& lpos, unsigned transform_index)
{
    lpos.clear(); 

    assert( m_posi->getDimensions() == 3 ); 
    unsigned ni = m_posi->getShape(0);  
    unsigned nj = m_posi->getShape(1);
    unsigned nk = m_posi->getShape(2);
    assert( nk == 4 ); 
    unsigned count(0);  

    glm::mat4 itr = m_itransforms->getMat4(transform_index - 1 ); 

    LOG(debug) 
        << " transform_index " << transform_index
        << " itr " << glm::to_string( itr )
        ;  

    for(unsigned i=0 ; i < ni ; i++){
    for(unsigned j=0 ; j < nj ; j++){

        unsigned pxid = m_posi->getUInt(i,j,3,0) ; 
        unsigned gc = pxid >> 24 ;         // geocode for pixel
        unsigned ti = pxid & 0xffffff ;    // transform index for pixel
        bool select = ti == transform_index  ;  
        if(!select) continue ;   

        assert( gc > 0 ); 
        count++ ;

        glm::vec4 posi = m_posi->getQuad(i,j,0); 
        posi.w = 1.f ;      // scrub the identity 
        
        glm::vec4 lpo = itr * posi ;   // transform global 3d coordinate into local frame 
        LOG(debug) 
            << " posi " << glm::to_string( posi )  
            << " lpo " << glm::to_string( lpo )
            ;  
 
        lpos.push_back(lpo); 
    }
    }
    LOG(debug) 
        << " transform_index " << transform_index
        << " lpos " << lpos.size() 
        ;
}

