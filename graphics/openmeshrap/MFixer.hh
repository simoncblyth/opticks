#pragma once
#include <cstddef>

class GGeo ; 
class MTool ; 

class MFixer {
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

inline MFixer::MFixer(GGeo* ggeo) : 
    m_ggeo(ggeo), 
    m_tool(NULL),
    m_verbose(false)
{
    init();
}

inline void MFixer::setVerbose(bool verbose)
{
    m_verbose = verbose ; 
}

