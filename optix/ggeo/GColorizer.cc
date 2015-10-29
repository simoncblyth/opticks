#include "GColorizer.hh"
#include "GVector.hh"
#include "GGeo.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GItemIndex.hh"
#include "GBoundary.hh"
#include "GColors.hh"

#include <iomanip>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void GColorizer::traverse()
{
    if(!m_target)
    {
        LOG(warning) << "GColorizer::traverse must setTarget before traverse " ;
        return ;  
    }
    GSolid* root = m_ggeo->getSolid(0);
    traverse(root, 0);
    LOG(info) << "GColorizer::traverse colorized nodes " << m_num_colorized ; 
}

void GColorizer::traverse( GNode* node, unsigned int depth)
{
    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GMesh* mesh = solid->getMesh();
    unsigned int nvert = mesh->getNumVertices();

    bool selected = solid->isSelected() && solid->getRepeatIndex() == m_repeat_index ;
    if(selected)
    {
        if( m_style == SURFACE_INDEX )
        { 
            gfloat3* surfcolor = getSurfaceColor( node );
            if(surfcolor) 
            {
                ++m_num_colorized ; 
                for(unsigned int i=0 ; i<nvert ; ++i ) m_target[m_cur_vertices+i] = *surfcolor ; 
            }
            delete surfcolor ; 
        } 
        else if( m_style == PSYCHEDELIC_VERTEX || m_style == PSYCHEDELIC_NODE || m_style == PSYCHEDELIC_MESH )  // every VERTEX/SOLID/MESH a different color 
        {
            assert(m_colors);
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
                m_target[m_cur_vertices+i] = m_colors->getPsychedelic(index) ; 
            } 
        }

        m_cur_vertices += nvert ;      // offset within the flat arrays
    }
    for(unsigned int i = 0; i < node->getNumChildren(); i++) traverse(node->getChild(i), depth + 1 );
}


gfloat3* GColorizer::getSurfaceColor(GNode* node)
{
    if(!m_surfaces) return NULL ; 

    gfloat3* nodecolor(NULL) ; 
    GSolid* solid = dynamic_cast<GSolid*>(node) ;
    GBoundary* boundary = solid->getBoundary();
    GPropertyMap<float>* imat = boundary->getInnerMaterial() ;
    GPropertyMap<float>* omat = boundary->getOuterMaterial() ;
    GPropertyMap<float>* isur = boundary->getInnerSurface() ;
    GPropertyMap<float>* osur = boundary->getOuterSurface() ;

    unsigned int colorcode(UINT_MAX) ; 

    if(isur->hasDefinedName() && m_surfaces->hasItem(isur->getShortName()))
    {
         colorcode = m_surfaces->getColorCode(isur->getShortName());

         LOG(debug) << "GColorizer::getSurfaceColor " 
                    << " inner " << std::setw(25) << isur->getShortName() 
                    << " color " << std::hex << colorcode 
                     ;   
    }
    else if(osur->hasDefinedName() && m_surfaces->hasItem(osur->getShortName()))
    {
         colorcode = m_surfaces->getColorCode(osur->getShortName());

         LOG(debug) << "GColorizer::getSurfaceColor " 
                    << " outer " << std::setw(25) << osur->getShortName() 
                    << " color " << std::hex << colorcode 
                    ;
    }

    return colorcode == UINT_MAX ? NULL : GItemIndex::makeColor(colorcode) ; 
}


//
//  observations as change gl/nrm/vert.glsl and vertex colors
//
//  initially without flipping normals had to fine tune light positions
//  in order to see anything and everything was generally very dark
//
//  with normals flipped things become much more visible, which makes
//  sense given the realisation that the "canonical" lighting situation 
//  is from inside geometry, which togther with outwards normals
//  means that have to flip the normals to see something 
//     

