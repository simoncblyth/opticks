#include "GSolid.hh"

#include "stdio.h"

GSolid::GSolid( GMesh* mesh, GMaterial* material1, GMaterial* material2, GSurface* surface1, GSurface* surface2 )
         : 
         m_mesh(mesh),
         m_material1(material1),
         m_material2(material2),
         m_surface1(surface1),
         m_surface2(surface2)
{
    // NB not taking ownership yet 
}

GSolid::~GSolid()
{
}

void GSolid::Summary(const char* msg )
{
   printf("%s\n", msg );
}


 
