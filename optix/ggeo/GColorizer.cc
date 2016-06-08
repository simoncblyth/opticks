#include "GColorizer.hh"

#include "GVector.hh"

#include "GGeo.hh"
#include "GMesh.hh"
#include "GSolid.hh"
#include "GItemIndex.hh"
#include "GBndLib.hh"
#include "GSurfaceLib.hh"

#include <iomanip>

#include "OpticksAttrSeq.hh"
#include "OpticksColors.hh"
#include "NQuad.hpp"
#include "BLog.hh"
// trace/debug/info/warning/error/fatal

void GColorizer::init()
{
    m_blib = m_ggeo->getBndLib();
    m_slib = m_ggeo->getSurfaceLib();
    m_colors = m_ggeo->getColors();
}


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
            nvec3 surfcolor = getSurfaceColor( node );
            ++m_num_colorized ; 
            for(unsigned int i=0 ; i<nvert ; ++i ) m_target[m_cur_vertices+i] = surfcolor ; 
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



nvec3 GColorizer::getSurfaceColor(GNode* node)
{

    GSolid* solid = dynamic_cast<GSolid*>(node) ;

    unsigned int boundary = solid->getBoundary();

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

