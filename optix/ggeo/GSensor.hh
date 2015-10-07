#pragma once
#include "string.h"
#include <string>

/*

Currently are keeping one GSensor for each node with associated identifier...
This means there are multiple (5 in the below example)  GSensor with the same identifier

GSensorList::dump : 6888 sensors 
GSensor  index      0 idhex 1010101 iddec 16843009 node_index   3199 name /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:1#pvAdPmtUnit#pvAdPmt
GSensor  index      1 idhex 1010101 iddec 16843009 node_index   3200 name /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
GSensor  index      2 idhex 1010101 iddec 16843009 node_index   3201 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
GSensor  index      3 idhex 1010101 iddec 16843009 node_index   3202 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
GSensor  index      4 idhex 1010101 iddec 16843009 node_index   3203 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode

GSensor  index      5 idhex 1010102 iddec 16843010 node_index   3205 name /dd/Geometry/AD/lvOIL#pvAdPmtArray#pvAdPmtArrayRotated#pvAdPmtRingInCyl:1#pvAdPmtInRing:2#pvAdPmtUnit#pvAdPmt
GSensor  index      6 idhex 1010102 iddec 16843010 node_index   3206 name /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum
GSensor  index      7 idhex 1010102 iddec 16843010 node_index   3207 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode
GSensor  index      8 idhex 1010102 iddec 16843010 node_index   3208 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiBottom
GSensor  index      9 idhex 1010102 iddec 16843010 node_index   3209 name /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiDynode

*/


class GSensor {
   public:
       static unsigned int RefIndex(GSensor* sensor);
       static const unsigned int UNSET_INDEX ;   

   public:
       GSensor( unsigned int index, unsigned int id, const char* node_name, unsigned int node_index  );

   public:
       unsigned int getIndex();      // 0 based sensor index in node tree order
       unsigned int getIndex1();     // 1 based sensor index in node tree order
       unsigned int getId();
       const char*  getNodeName();
       unsigned int getNodeIndex();

       std::string description();

   private:
       unsigned int m_index ;
       unsigned int m_id ;
       const char*  m_node_name ; 
       unsigned int m_node_index ;
};

inline GSensor::GSensor(unsigned int index, unsigned int id, const char* node_name, unsigned int node_index)
       :
       m_index(index),
       m_id(id),
       m_node_name(strdup(node_name)),
       m_node_index(node_index)
{
}

inline unsigned int GSensor::getIndex()
{
    return m_index ; 
}
inline unsigned int GSensor::getIndex1()
{
    return m_index + 1 ; 
}

inline unsigned int GSensor::RefIndex(GSensor* sensor)
{
    return sensor ? sensor->getIndex1() : GSensor::UNSET_INDEX  ;
}


inline unsigned int GSensor::getNodeIndex()
{
    return m_node_index ; 
}
inline unsigned int GSensor::getId()
{
    return m_id ; 
}
inline const char* GSensor::getNodeName()
{
    return m_node_name ; 
}



