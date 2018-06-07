#include "X4Material.hh"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

#include "X4Material.hh"
#include "G4Material.hh"
#include "NPY.hpp"
#include "PLOG.hh"

X4Material::X4Material( const G4Material* material ) 
   :
   m_material(material)
{
   init() ;
}

void X4Material::init()
{
}

std::string X4Material::desc() const 
{
    std::stringstream ss ; 
    ss << "X4Material"
       ;
    return ss.str();
}


