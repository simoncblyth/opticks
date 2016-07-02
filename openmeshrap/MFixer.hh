#pragma once

class GGeo ; 
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
       GGeo*   m_ggeo ;
       MTool*  m_tool ; 
       bool    m_verbose ; 

};


