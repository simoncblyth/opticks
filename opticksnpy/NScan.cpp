
#include <deque>
#include <iostream>
#include <sstream>
#include "PLOG.hh"

#include "GLMFormat.hpp"

#include "NNode.hpp"
#include "NGLMExt.hpp"

#include "NScanLine.hpp"
#include "NScan.hpp"



const std::string& NScan::get_message() const 
{
    return m_message ; 
}




NScan::NScan( const nnode& node, unsigned verbosity ) 
    :
    m_node(node),
    m_bbox(node.bbox()),
    m_verbosity(verbosity),
    m_message("")
{
    init();
}

void NScan::init()
{
    init_cage( m_bbox, m_bmin, m_bmax, m_bcen, 0.1f );

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
    LOG(info) << desc() ; 

    if(m_verbosity > 3)
    m_node.dump_full("NScan::init");
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


void NScan::init_cage(const nbbox& bb, glm::vec3& bmin, glm::vec3& bmax, glm::vec3& bcen, float sidescale  ) // static
{
    bmin.x = bb.min.x - bb.side.x*sidescale ;
    bmin.y = bb.min.y - bb.side.y*sidescale ;
    bmin.z = bb.min.z - bb.side.z*sidescale ;

    bmax.x = bb.max.x + bb.side.x*sidescale ;
    bmax.y = bb.max.y + bb.side.y*sidescale ;
    bmax.z = bb.max.z + bb.side.z*sidescale ;

    bcen.x =  (bb.min.x + bb.max.x)/2.f ;
    bcen.y =  (bb.min.y+bb.max.y)/2.f ;
    bcen.z =  (bb.min.z+bb.max.z)/2.f ;

}


unsigned NScan::autoscan()
{
    if(m_verbosity > 0)
    LOG(info) << "NScan::autoscan" 
              << " verbosity " << m_verbosity 
               ;
   
    // center x,y, -z->z

    glm::vec3 beg( m_bcen.x, m_bcen.y, int(m_bmin.z) );  // trunc to integer mm in z 
    glm::vec3 end( m_bcen.x, m_bcen.y, int(m_bmax.z) );
    glm::vec3 step( 0,0,1 ) ;

    NScanLine line(beg, end, step, m_verbosity );

    if(m_verbosity > 1)
    std::cout << line.desc() << std::endl ; 

    line.setNodes(m_nodes);

    line.find_zeros();

    unsigned nzero = line.count_zeros(0);

    const std::string& msg = line.get_message();
    if(!msg.empty())  m_message.assign(msg);


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

