#pragma once

#include <map>
#include <string>

/*

*GColorMap* manages the association of named items with colors
 
Example of usage from GLoader::

    m_flags->setColorMap(GColorMap::load("$HOME/.opticks", "GFlagIndexColors.json"));      
    m_materials->setColorMap(GColorMap::load("$HOME/.opticks", "GMaterialIndexColors.json"));
    m_surfaces->setColorMap(GColorMap::load("$HOME/.opticks", "GSurfaceIndexColors.json"));
  
simon:.opticks blyth$ cat GFlagIndexColors.json 
{
    "CERENKOV":"white",
    "SCINTILLATION":"white",
    "TORCH":"white",
    "MISS":"grey",
    "BULK_ABSORB":"red",
    "BULK_REEMIT":"green", 
    "BULK_SCATTER":"blue",    
    "SURFACE_DETECT":"purple",
    "SURFACE_ABSORB":"orange",      
     ...


*/

class GColorMap  {
   public:
       static GColorMap* load(const char* dir, const char* name);
   public:
       GColorMap();
       void dump(const char* msg="GColorMap::dump");

   public:
       void addItemColor(const char* iname, const char* color); 
       const char* getItemColor(const char* iname, const char* missing=NULL);

   private:
       void loadMaps(const char* idpath, const char* name);

   private:
       std::map<std::string, std::string>  m_iname2color ; 

};


inline GColorMap::GColorMap()
{
}

inline void GColorMap::addItemColor(const char* iname, const char* color)
{
     m_iname2color[iname] = color ; 
}

inline const char* GColorMap::getItemColor(const char* iname, const char* missing)
{
     // hmm maybe string data moved around as item/color are added to map, 
     // so best to query only after the map is completed for consistent pointers
     return m_iname2color.count(iname) == 1 ? m_iname2color[iname].c_str() : missing ; 
}


