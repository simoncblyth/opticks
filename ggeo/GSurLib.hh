#pragma once

/**
GSurLib
==========

PERHAPS MIS-PLACED IT BELONGS IN cfg4- 

See::

    GSurLibTest --dbgsurf


To understand GSurLib it is best to first study its primary 
user CSurLib::convert noting the ingredients required 
to create the G4 surfaces.

GSurLib and the contained GSur are created post-cache
as a constituent of GGeo in GGeo::createSurLib.
This creation is deferred until GGeo::getSurLib which 
normally occurs at CG4/CGeometry/CSurLib instanciation

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

*assignType*
     invokes GSur::assignType on each instance, setting type for U for unused surfaces
     Observe that the "perfect*" surfaces that are often used with test geometry
     do not appear in mesh0 so they all get set to type U for unused. 

The original intention was to distinguish skin from border
post-cache heuristically from observation of the characteristics of the 
surfaces of all solids. 
However it turns out not be be possible to make such a determination, 
so a cheat based on names of bordersurfaces grepped from the .dae is used.


TODO: equivalents for GGeoTest(NCSG) and GScene(GDML2GLTF) geometries
------------------------------------------------------------------------

Can GBndLib distinguish just by osur&isur OR osur^isur ? 

G4LogicalSkinSurface
    surface surrounding a single logical volume
    ("directionless" equal action from either side) 

    * opticks:  isur && osur && (isur == osur)

G4LogicalBorderSurface
    surfaces defined by the boundary of two physical volumes
    ("directional" : action only in prescribed pv order)

    * opticks:  (!!isur)^(!!osur) 


Opticks boundary indices are ascribed onto GSolid/GNode 
(ie at tree level) so can definitely hold the G4 model.
The question is more on the details of implementing it 
and conversion to it.

Current GBndLib has no cases of both osur and isur, so
all surfaces are border with directionality and none are 
skin.  

Perhaps GSurLib was used for fix this up by cheating with names.

GGeoTest(NCSG)
~~~~~~~~~~~~~~~~

Expanding to test geometry : primary remit is to enable the expression 
in G4 optical surfaces the same intent as implemented 
in Opticks surfaces.

* distinguishing border/skin is clear from the GBndLib 

  * actually the distinction often mute for the common perfectAbsorbSurface, 
    because typically never get photons from outside 


GScene(GDML2GLTF)
~~~~~~~~~~~~~~~~~~~~~




**/

#include <set>
#include <vector>
#include <string>
#include <utility>

class Opticks ; 

class GGeoBase ;
class GGeoLib ;
class GNodeLib ;
class GSur ; 
class GSurfaceLib ; 
class GBndLib ; 

#include "GGEO_API_EXPORT.hh"

class GGEO_API GSurLib 
{
         friend class CDetector ; 
    public:
         static const unsigned UNSET ;  
         static void pushBorderSurfacesDYB(std::vector<std::string>& names);
         bool        isBorderSurface(const char* name);
    public:
         GSurLib(Opticks* ok, GGeoBase* ggb);
         void close();   // invoked by CDetector after potential mesh mods have been made

    public:
         std::string desc() const ; 
         void dump(const char* msg="GSurLib::dump");
         GSurfaceLib* getSurfaceLib();
         Opticks*     getOpticks();

    public:
         unsigned     getNumSur() const ;
         GSur*        getSur(unsigned index);

         void         getSurfacePair(std::pair<GSur*,GSur*>& osur_isur, unsigned boundary);
         bool         isClosed();
    private:
         void init();
         void collectSur();


         void examineSolidBndSurfaces();
         void assignType();
         void add(GSur* surf);

         std::string desc(const std::set<unsigned>& bnd);

    public:
         Opticks*              m_ok ; 
         GGeoLib*              m_geolib ;
         GNodeLib*             m_nodelib ;
         GBndLib*              m_blib ; 
         GSurfaceLib*          m_slib ; 

         bool                  m_is_test ; 
         bool                  m_dbgsurf ; 
         bool                  m_closed ; 

         std::vector<GSur*>        m_surs ; 
         std::vector<std::string>  m_bordersurface ; 

};


