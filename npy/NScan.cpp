
#include <deque>
#include <iostream>
#include <sstream>
#include "PLOG.hh"

#include "SSys.hh"

#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NGLMExt.hpp"

#include "NScanLine.hpp"
#include "NScan.hpp"



bool NScan::has_message() const 
{
    return !m_message.empty() ; 
}

const std::string& NScan::get_message() const 
{
    return m_message ; 
}

unsigned NScan::get_nzero() const 
{
    return m_nzero ; 
}
void NScan::set_nzero(unsigned nzero) 
{
    m_nzero = nzero ; 
}


NScan::NScan( const nnode& node, unsigned verbosity ) 
    :
    m_node(node),
    m_bbox(node.bbox()),
    m_verbosity(verbosity),
    m_message(""),
    m_nzero(0)
{
    init();
}

void NScan::init()
{
    float sidescale = 0.1f ; 
    float minmargin = 2.f ;  // mm

    init_cage( m_bbox, m_bmin, m_bmax, m_bcen, sidescale, minmargin, m_verbosity );

    m_node.collect_prim(m_prim);    // recursive collection of list of all primitives in tree

    m_nodes.push_back( &m_node) ; 
    m_node.collect_prim(m_nodes);  // collect root and primitives into m_nodes

    m_num_prim = m_prim.size();
    m_num_nodes = m_nodes.size();

    for(unsigned i=0 ; i < m_num_nodes ; i++)
    {
        const nnode* n = m_nodes[i] ; 
        std::string  t = n->tag(); 
        m_tags.push_back(t) ;
    }


    if(m_verbosity > 1)
    {
        LOG(info) << desc() ; 
    }

    if(m_verbosity > 3)
    m_node.dump();
}

std::string NScan::desc() const 
{
    std::stringstream ss ; 
    ss << "NScan"
       << " num_prim " << m_num_prim 
       << " num_nodes " << m_num_nodes 
       ; 

    for(unsigned i=0 ; i < m_num_nodes ; i++) ss << " (" << m_tags[i] << ")" ; 

    ss  << std::endl  
        << gpresent("bmin", m_bmin) << std::endl 
        << gpresent("bmax", m_bmax) << std::endl 
        << gpresent("bcen", m_bcen) << std::endl 
        ;

    return ss.str();
}


void NScan::init_cage(const nbbox& bb, glm::vec3& bmin, glm::vec3& bmax, glm::vec3& bcen, float sidescale, float minmargin, unsigned verbosity ) // static
{
    glm::vec3 delta(0,0,0);

    // prevent cage margin from being too small for objs that are thin along some axis

    glm::vec3 bb_side = bb.side();

    delta.x = std::max<float>(bb_side.x*sidescale, minmargin) ;
    delta.y = std::max<float>(bb_side.y*sidescale, minmargin) ;
    delta.z = std::max<float>(bb_side.z*sidescale, minmargin) ;

    bmin.x = bb.min.x - delta.x ;
    bmin.y = bb.min.y - delta.y ;
    bmin.z = bb.min.z - delta.z ;

    bmax.x = bb.max.x + delta.x ;
    bmax.y = bb.max.y + delta.y ;
    bmax.z = bb.max.z + delta.z ;

    bcen.x =  (bb.min.x + bb.max.x)/2.f ;
    bcen.y =  (bb.min.y+bb.max.y)/2.f ;
    bcen.z =  (bb.min.z+bb.max.z)/2.f ;


    if(verbosity > 3)
    {
        LOG(info) << "NScan::init_cage"
                  << " verbosity " << verbosity
                  << " sidescale " << sidescale
                  << " minmargin " << minmargin
                  ; 

        std::cout << "delta " << glm::to_string(delta) << std::endl ; 
        std::cout << "bmin  " << glm::to_string(bmin) << std::endl ; 
        std::cout << "bmax  " << glm::to_string(bmax) << std::endl ; 
        std::cout << "bcen  " << glm::to_string(bcen) << std::endl ; 
    }





}


unsigned NScan::autoscan(float mmstep)
{
    if(m_verbosity > 0)
    {
    LOG(info) << "NScan::autoscan" 
              << " verbosity " << m_verbosity 
              << " mmstep " << mmstep
               ;
    }
   
    // center x,y, -z->z

    glm::vec3 beg( m_bcen.x, m_bcen.y, int(m_bmin.z) );  // trunc to integer mm in z 
    glm::vec3 end( m_bcen.x, m_bcen.y, int(m_bmax.z) );
    glm::vec3 step( 0,0, mmstep ) ;

    NScanLine line(beg, end, step, m_verbosity );

    if(m_verbosity > 1)
    std::cout << line.desc() << std::endl ; 

    line.setNodes(m_nodes);

    line.find_zeros();

    unsigned nzero = line.count_zeros(0);

    const std::string& msg = line.get_message();
    if(!msg.empty())  m_message.assign(msg);


    if(m_verbosity > 2)
    {
        if(SSys::IsVERBOSE())
        {
            line.dump("VERBOSE full dump");
        }
        else
        { 
            unsigned step_window = 10 ; 
            if(nzero != 2)
            {
                line.dump_zeros(0, step_window);
            }
        }
    }

    set_nzero(nzero);

    //line.dump("NScan::autoscan", 300, 330);
    return nzero ; 
}





void NScan::scan(std::vector<float>& sd, const glm::vec3& origin, const glm::vec3& direction, const glm::vec3& step )
{
    LOG(info) << "NScan::scan" ;
    sd.clear(); 

    glm::vec3 end = origin + direction ; 

    NScanLine line(origin, end, step, m_verbosity );
    line.setNodes(m_nodes);
    line.dump("NScan::scan");

}

