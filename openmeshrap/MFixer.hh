#pragma once

class GGeo ; 
class GMeshLib ; 
class MTool ; 

#include "MESHRAP_API_EXPORT.hh"
class MESHRAP_API MFixer {
   public:
       MFixer(GGeo* ggeo);
       void setVerbose(bool verbose=true);
       void fixMesh();
   private:
       void init();
   private:
       GGeo*     m_ggeo ;
       GMeshLib* m_meshlib ; 
       MTool*    m_tool ; 
       bool      m_verbose ; 

};


