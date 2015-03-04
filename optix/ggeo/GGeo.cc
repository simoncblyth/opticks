#include "GGeo.hh"

#include "GSkinSurface.hh"
#include "GBorderSurface.hh"
#include "GMaterial.hh"
#include "GSolid.hh"
#include "GMesh.hh"

#include "stdio.h"

#define BSIZ 50

GGeo::GGeo()
{
   printf("GGeo::GGeo\n");
}

GGeo::~GGeo()
{
}

void GGeo::Summary(const char* msg)
{
    printf("%s ms %lu so %lu mt %lu bs %lu ss %lu \n", msg, m_meshes.size(), m_solids.size(), m_materials.size(), m_border_surfaces.size(), m_skin_surfaces.size() );  

    char mbuf[BSIZ];

    for(unsigned int ims=0 ; ims < m_meshes.size()  ; ims++ )
    {
        GMesh* ms = m_meshes[ims];
        snprintf(mbuf,BSIZ, "%s ms %u", msg, ims);
        ms->Summary(mbuf);
    }

    for(unsigned int ibs=0 ; ibs < m_border_surfaces.size()  ; ibs++ )
    {
        GBorderSurface* bs = m_border_surfaces[ibs];
        snprintf(mbuf,BSIZ, "%s bs %u", msg, ibs);
        bs->Summary(mbuf);
    }

    for(unsigned int iss=0 ; iss < m_skin_surfaces.size()  ; iss++ )
    {
        GSkinSurface* ss = m_skin_surfaces[iss];
        snprintf(mbuf,BSIZ, "%s ss %u", msg, iss);
        ss->Summary(mbuf);
    }
    for(unsigned int imat=0 ; imat < m_materials.size()  ; imat++ )
    {
        GMaterial* mat = m_materials[imat];
        snprintf(mbuf,BSIZ, "%s mt %u", msg, imat);
        mat->Summary(mbuf);
    }
    for(unsigned int isol=0 ; isol < m_solids.size()  ; isol++ )
    {
        GSolid* sol = m_solids[isol];
        snprintf(mbuf,BSIZ, "%s so %u", msg, isol);
        sol->Summary(mbuf);
    }
}


void GGeo::add(GMesh* mesh)
{
    m_meshes.push_back(mesh);
}

void GGeo::add(GSolid* solid)
{
    m_solids.push_back(solid);
}

void GGeo::add(GMaterial* material)
{
    m_materials.push_back(material);
}

void GGeo::add(GBorderSurface* surface)
{
    m_border_surfaces.push_back(surface);
}

void GGeo::add(GSkinSurface* surface)
{
    m_skin_surfaces.push_back(surface);
}



unsigned int GGeo::getNumMeshes()
{
    return m_meshes.size();
}
unsigned int GGeo::getNumSolids()
{
    return m_solids.size();
}
unsigned int GGeo::getNumMaterials()
{
    return m_materials.size();
}
unsigned int GGeo::getNumBorderSurfaces()
{
    return m_border_surfaces.size();
}
unsigned int GGeo::getNumSkinSurfaces()
{
    return m_skin_surfaces.size();
}



GMesh* GGeo::getMesh(unsigned int index)
{
    return m_meshes[index];
}  
GMaterial* GGeo::getMaterial(unsigned int index)
{
    return m_materials[index];
}
GSolid* GGeo::getSolid(unsigned int index)
{
    return m_solids[index];
}
GSkinSurface* GGeo::getSkinSurface(unsigned int index)
{
    return m_skin_surfaces[index];
}
GBorderSurface* GGeo::getBorderSurface(unsigned int index)
{
    return m_border_surfaces[index];
}






