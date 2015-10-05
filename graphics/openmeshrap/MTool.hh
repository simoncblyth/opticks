#pragma once

#include <string>
class GMesh ; 
class GCache ; 

class MTool {
   public:
       MTool();
       static GMesh* joinSplitUnion(GMesh* mesh, GCache* config);
   public:
       unsigned int countMeshComponents(GMesh* gm);
   public:
       std::string& getOut();
       std::string& getErr();
       unsigned int getNoise();
   private:
       unsigned int countMeshComponents_(GMesh* gm);

   private:
       std::string  m_out ; 
       std::string  m_err ;
       unsigned int m_noise ;  

};


inline MTool::MTool() : m_noise(0) 
{
}

inline std::string& MTool::getOut()
{
    return m_out ; 
}
inline std::string& MTool::getErr()
{
    return m_err ; 
}
inline unsigned int MTool::getNoise()
{
    return m_noise ; 
}





