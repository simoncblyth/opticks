#pragma once

template <typename T> class NPY ;
class NTrianglesNPY ; 

#include <vector>
#include <cstddef>
#include <glm/fwd.hpp>

class Opticks ; 
#include "OpticksCSG.h"

class NCSG ; 
struct gbbox ; 
struct nnode ; 

class GBndLib ; 
class GMeshLib ; 
class GVolume ; 
class GMesh ; 

/**

GMaker
=======

Only one canonical instance m_maker resides in GGeoTest 


**/

#include "GGEO_API_EXPORT.hh"
class GGEO_API GMaker {
       friend class GMakerTest ; 
    public:
        static std::string PVName(const char* shapename, int idx=-1);
        static std::string LVName(const char* shapename, int idx=-1);
   public:
       GMaker(Opticks* ok, GBndLib* blib, GMeshLib* meshlib );
   public:
       GVolume* make(unsigned int index, OpticksCSG_t typecode, glm::vec4& param, const char* spec);
   public:
       GMesh*   makeMeshFromCSG( NCSG* csg ) ; 
       GVolume* makeFromMesh( const GMesh* mesh ) const ; 
   private:
       void init();    

       static GVolume* makePrism(glm::vec4& param, const char* spec);
       static GVolume* makeBox(glm::vec4& param);
       static GVolume* makeZSphere(glm::vec4& param);
       static GVolume* makeZSphereIntersect_DEAD(glm::vec4& param, const char* spec);
       static void makeBooleanComposite(char shapecode, std::vector<GVolume*>& volumes,  glm::vec4& param, const char* spec);
       static GVolume* makeBox(gbbox& bbox);
   private:
       static GVolume* makeSubdivSphere(glm::vec4& param, unsigned int subdiv=3, const char* type="I");
       static NTrianglesNPY* makeSubdivSphere(unsigned int nsubdiv=3, const char* type="I");
       static GVolume* makeSphere(NTrianglesNPY* tris);
   private:
       Opticks*  m_ok ; 
       GBndLib*  m_bndlib ; 
       GMeshLib* m_meshlib ; 
};


