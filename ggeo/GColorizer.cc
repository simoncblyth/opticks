#include <cstddef>
#include <iomanip>

#include "OpticksAttrSeq.hh"
#include "OpticksColors.hh"
#include "NQuad.hpp"

#include "GVector.hh"
#include "GMesh.hh"
#include "GMergedMesh.hh"
#include "GVolume.hh"
#include "GItemIndex.hh"

#include "GNodeLib.hh"
#include "GGeoLib.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"
#include "GColorizer.hh"

#include "PLOG.hh"
// trace/debug/info/warning/error/fatal

const plog::Severity GColorizer::LEVEL = debug ; 


GColorizer::GColorizer(GNodeLib* nodelib, GGeoLib* geolib, GBndLib* blib, OpticksColors* colors, GColorizer::Style_t style ) 
    :
    m_target(NULL),
    m_nodelib(nodelib),
    m_geolib(geolib),
    m_blib(blib),
    m_slib(blib->getSurfaceLib()),
    m_colors(colors),
    m_style(style),
    m_cur_vertices(0),
    m_num_colorized(0),
    m_repeat_index(0)
{
     init();
}

void GColorizer::init()
{
    if(!m_colors)
         LOG(warning) << "GColorizer::init m_colors NULL " ; 
}

void GColorizer::setTarget(nvec3* target)
{
    m_target =  target ; 
}
void GColorizer::setRepeatIndex(unsigned ridx)
{
    m_repeat_index = ridx ; 
}

void GColorizer::writeVertexColors()
{
    GMergedMesh* mesh0 = m_geolib->getMergedMesh(0);
    GVolume* root = m_nodelib->getVolume(0);
    writeVertexColors( mesh0, root );
}

void GColorizer::writeVertexColors(GMergedMesh* mesh0, GVolume* root)
{
    assert(mesh0);

    gfloat3* vertex_colors = mesh0->getColors();

    setTarget( reinterpret_cast<nvec3*>(vertex_colors) );
    setRepeatIndex(mesh0->getIndex()); 

    traverse(root);
}


/**
GColorizer::traverse
----------------------

Visits all vertices of selected volumes setting the 
vertex colors of the GMergedMesh based on indices of
objects configured via the style.

**/

void GColorizer::traverse(GVolume* root)
{
    if(!m_target)
    {
        LOG(fatal) << "GColorizer::traverse must setTarget before traverse " ;
        return ;  
    }
    LOG(LEVEL) << "GColorizer::traverse START" ; 

    traverse_r(root, 0);

    LOG(LEVEL) << "GColorizer::traverse colorized nodes " << m_num_colorized ; 
}

void GColorizer::traverse_r( GNode* node, unsigned depth)
{
    GVolume* volume = dynamic_cast<GVolume*>(node) ;
    const GMesh* mesh = volume->getMesh();
    unsigned nvert = mesh->getNumVertices();


    bool selected = volume->isSelected() && volume->getRepeatIndex() == m_repeat_index ;

    LOG(verbose) << "GColorizer::traverse"
              << " depth " << depth
              << " node " << ( node ? node->getIndex() : 0 )
              << " nvert " << nvert
              << " selected " << selected
              ; 


    if(selected)
    {
        if( m_style == SURFACE_INDEX )
        { 
            nvec3 surfcolor = getSurfaceColor( node );
            ++m_num_colorized ; 
            for(unsigned int i=0 ; i<nvert ; ++i ) m_target[m_cur_vertices+i] = surfcolor ; 
        } 
        else if( m_style == PSYCHEDELIC_VERTEX || m_style == PSYCHEDELIC_NODE || m_style == PSYCHEDELIC_MESH )  // every VERTEX/SOLID/MESH a different color 
        {
            for(unsigned int i=0 ; i<nvert ; ++i ) 
            {
                unsigned int index ; 
                switch(m_style)
                {
                    case PSYCHEDELIC_VERTEX : index = i                ;break; 
                    case PSYCHEDELIC_NODE   : index = node->getIndex() ;break; 
                    case PSYCHEDELIC_MESH   : index = mesh->getIndex() ;break; 
                    default                 : index = 0                ;break; 
                }

                if(m_colors)
                { 
                    m_target[m_cur_vertices+i] = m_colors->getPsychedelic(index) ;
                }
      
            } 
        }

        m_cur_vertices += nvert ;      // offset within the flat arrays
    }
    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse_r(node->getChild(i), depth + 1 );
}



nvec3 GColorizer::getSurfaceColor(GNode* node)
{

    GVolume* volume = dynamic_cast<GVolume*>(node) ;

    unsigned int boundary = volume->getBoundary();

    guint4 bnd = m_blib->getBnd(boundary);
    unsigned int isur_ = bnd.z ;  
    unsigned int osur_ = bnd.w ;  

    const char* isur = m_slib->getName(isur_);
    const char* osur = m_slib->getName(osur_);

    unsigned int colorcode(UINT_MAX) ; 
    if(isur)
    {
        colorcode = m_slib->getAttrNames()->getColorCode(isur);    
    } 
    else if(osur)
    {
        colorcode = m_slib->getAttrNames()->getColorCode(osur);    
    }  

    if(colorcode != UINT_MAX )
    LOG(debug) << "GColorizer::getSurfaceColor " 
              << " isur " << std::setw(3) << isur_ << std::setw(30) <<  ( isur ? isur : "-" )
              << " osur " << std::setw(3) << osur_ << std::setw(30) <<  ( osur ? osur : "-" )
              << " colorcode " << std::hex << colorcode  << std::dec 
              ; 

   return colorcode == UINT_MAX ? make_nvec3(1,0,1) : GItemIndex::makeColor(colorcode) ; 
}


