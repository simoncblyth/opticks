#include <glm/glm.hpp>
#include "NGenerator.hpp"


NGenerator::NGenerator(const nbbox& bb)
   :
   m_bb(bb),
   m_side({bb.max.x - bb.min.x, bb.max.y - bb.min.y, bb.max.z - bb.min.z }), 
   m_dist(0.f, 1.f),
   m_gen(m_rng, m_dist)
{
}

void NGenerator::operator()(nvec3& p)
{
    p.x = m_bb.min.x + m_gen()*m_side.x ; 
    p.y = m_bb.min.y + m_gen()*m_side.y ; 
    p.z = m_bb.min.z + m_gen()*m_side.z ;
}

void NGenerator::operator()(glm::vec3& p)
{
    p.x = m_bb.min.x + m_gen()*m_side.x ; 
    p.y = m_bb.min.y + m_gen()*m_side.y ; 
    p.z = m_bb.min.z + m_gen()*m_side.z ;
}


