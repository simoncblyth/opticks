#pragma once

/**
GSurLib
==========

GSurLib and the contained GSur are created post-cache
as a constituent of GGeo in GGeo::loadFromCache.
GSurLib facilitates recreation of G4LogicalBorderSurface and G4LogicalSkinSurface 
instances that are missing from the CGDMLDetector.

At instanciation in GSurLib::init:

*collectSur*
     instanciate GSur wrapper objects containing each GPropertyMap<float> surface
     holding the border/skin surface classification 

*examineSolidBndSurfaces*
     loop over the solids of the global mesh0 examining the boundaries 
     of the solids and possible associated surfaces.  Using parent-child 
     nodeinfo stored for each solid determine physical volume pairs 
     and logical volumes for each of the surfaces, record these into the GSur  

     TODO: get this to work with test geometry... 


*assignType*
     invokes GSur::assignType on each instance, setting type for U for unused surfaces
     Observe that the "perfect*" surfaces that are often used with test geometry
     do not appear in mesh0 so they all get set to type U for unused. 

The original intention was to distinguish skin from border
post-cache heuristically from observation of the characteristics of the 
surfaces of all solids. 
However it turns out not be be possible to make such a determination, 
so a cheat based on names of bordersurfaces grepped from the .dae is used.

**/

#include <set>
#include <vector>

class GGeo ;
class GMergedMesh ;
class GSur ; 
class GSurfaceLib ; 
class GBndLib ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GSurLib 
{
    public:
         static const unsigned UNSET ;  
         static void pushBorderSurfaces(std::vector<std::string>& names);
         bool isBorderSurface(const char* name);
    public:
         GSurLib(GGeo* gg);
         void dump(const char* msg="GSurLib::dump");
         GSurfaceLib* getSurfaceLib();

         unsigned getNumSur();
         GSur* getSur(unsigned index);
    private:
         void init();
         void collectSur();
         void examineSolidBndSurfaces();
         void assignType();
         void add(GSur* surf);
         std::string desc(const std::set<unsigned>& bnd);
    public:
         GGeo*                 m_ggeo ;
         GMergedMesh*          m_mesh0 ; 
         GSurfaceLib*          m_slib ; 
         GBndLib*              m_blib ; 

         std::vector<GSur*>        m_surs ; 
         std::vector<std::string>  m_bordersurface ; 

};


