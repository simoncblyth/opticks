#include <algorithm>   
#include <sstream>   

#include "X4SolidList.hh"

X4SolidList::X4SolidList()
{
}

void X4SolidList::addSolid(G4VSolid* solid)
{
    if(hasSolid(solid)) return ; 
    m_solidlist.push_back(solid); 
}
 
bool X4SolidList::hasSolid(G4VSolid* solid) const 
{
    return std::find(m_solidlist.begin(), m_solidlist.end(), solid) != m_solidlist.end()  ;
}

unsigned X4SolidList::getNumSolids() const 
{
    return m_solidlist.size() ; 
}

std::string X4SolidList::desc() const 
{
    std::stringstream ss ; 
    ss << "X4SolidList"
       << " NumSolids " << getNumSolids() 
       ;
    return ss.str(); 
}

 
