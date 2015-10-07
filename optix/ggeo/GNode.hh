#pragma once

#include <string>
#include <vector>
#include "string.h"
#include "GVector.hh"
#include "GMatrix.hh"

class GMesh ;

class GNode {
  public:
      GNode(unsigned int index, GMatrixF* transform, GMesh* mesh);
      virtual ~GNode();

  private:
      void init();

  public: 
      void Summary(const char* msg="GNode::Summary");

  public:
      void setParent(GNode* parent);
      void addChild(GNode* child);
      void setDescription(char* desc);
      void setName(const char* name);
      const char* getName();

  public:
     void setRepeatIndex(unsigned int index);
     unsigned int getRepeatIndex();  

  public:
      //
      // **boundary indices live on the node rather than the mesh**
      //
      // as there are a relatively small number of meshes and many nodes
      // that utilize them with different transforms
      //
      // normally a single boundary per-node but allow the 
      // possibility of compound boundary nodes, eg for combined meshes
      //

  public: 
      // setters duplicate the index to all the faces
      // allowing simple merging when flatten a tree 
      // of nodes into a single structure
      //
      void setBoundaryIndices(unsigned int boundary_index);
      void setSensorIndices(unsigned int sensor_index);
  private:
      void setNodeIndices(unsigned int index); 

  public: 
      void setBoundaryIndices(unsigned int* boundary_indices);
      void setSensorIndices(unsigned int* sensor_indices);

      std::vector<unsigned int>& getDistinctBoundaryIndices();
      void updateDistinctBoundaryIndices();

  public:
      unsigned int  getIndex();
      GNode*        getParent(); 
      GNode*        getChild(unsigned int index);
      unsigned int  getNumChildren();
      char*         getDescription();
      gfloat3*      getLow();
      gfloat3*      getHigh();
      GMesh*        getMesh();

  public:
      unsigned int* getNodeIndices();
      unsigned int* getBoundaryIndices();
      unsigned int* getSensorIndices();

  public:
     void updateBounds();
     void updateBounds(gfloat3& low, gfloat3& high );

  public:
      GMatrixF*     getTransform();  // global transform
      GMatrixF* getLevelTransform();  // immediate "local" node transform
      GMatrixF* getRelativeTransform(GNode* base);  // product of transforms from beneath base node

  public:
      void setLevelTransform(GMatrixF* ltransform);

  public:
      // *calculateTransform* 
      //       successfully duplicates the global transform of a node by calculating 
      //       the product of levelTransforms (ie single PV-LV transform)
      //       from ancestors + self
      // 
      //       This can be verified for all nodes within GTreeCheck 
      //
      //
      GMatrixF*            calculateTransform();  

  public:
      std::vector<GNode*>& getAncestors();
      std::vector<GNode*>& getProgeny();
      std::string&         getProgenyDigest();
      std::string&         getLocalDigest();
      unsigned int         getProgenyCount();
      unsigned int         getProgenyNumVertices();  // includes self when m_selfdigest is true
      GNode*               findProgenyDigest(const std::string& pdig) ;
      std::vector<GNode*>  findAllProgenyDigest(std::string& dig);

  private:
      std::string          meshDigest();
      std::string          localDigest();
      static std::string   localDigest(std::vector<GNode*>& nodes, GNode* extra=NULL);
      void collectProgeny(std::vector<GNode*>& progeny);
      void collectAllProgenyDigest(std::vector<GNode*>& match, std::string& dig);

  private:
      bool                m_selfdigest ; // when true getProgenyDigest includes self node 
      unsigned int        m_index ; 
      GNode*              m_parent ; 
      std::vector<GNode*> m_children ;
      char*               m_description ;

  private: 
      GMatrixF*           m_transform ; 
      GMatrixF*           m_ltransform ; 
      GMesh*              m_mesh ; 
      gfloat3*            m_low ; 
      gfloat3*            m_high ; 

  private: 
      unsigned int*       m_boundary_indices ;
      unsigned int*       m_sensor_indices ;
      unsigned int*       m_node_indices ;
      const char*         m_name ; 
  private: 
      std::string         m_local_digest ; 
      std::string         m_progeny_digest ; 
      std::vector<GNode*> m_progeny ; 
      std::vector<GNode*> m_ancestors ; 
      unsigned int        m_progeny_count ; 
      unsigned int        m_repeat_index ; 
      unsigned int        m_progeny_num_vertices ;
  private: 
      std::vector<unsigned int> m_distinct_boundary_indices ;

};



 


inline GNode::GNode(unsigned int index, GMatrixF* transform, GMesh* mesh) 
    :
    m_selfdigest(true),
    m_index(index), 
    m_parent(NULL),
    m_description(NULL),
    m_transform(transform),
    m_ltransform(NULL),
    m_mesh(mesh),
    m_low(NULL),
    m_high(NULL),
    m_boundary_indices(NULL),
    m_sensor_indices(NULL),
    m_node_indices(NULL),
    m_name(NULL),
    m_progeny_count(0),
    m_repeat_index(0),
    m_progeny_num_vertices(0)
{
    init();
}


inline gfloat3* GNode::getLow()
{
    return m_low ; 
}
inline gfloat3* GNode::getHigh()
{
    return m_high ; 
}
inline GMesh* GNode::getMesh()
{
   return m_mesh ;
}
inline GMatrixF* GNode::getTransform()
{
   return m_transform ;
}

inline unsigned int* GNode::getBoundaryIndices()
{
    return m_boundary_indices ; 
}
inline unsigned int* GNode::getNodeIndices()
{
    return m_node_indices ; 
}
inline unsigned int* GNode::getSensorIndices()
{
    return m_sensor_indices ; 
}



inline void GNode::setBoundaryIndices(unsigned int* boundary_indices)
{
    m_boundary_indices = boundary_indices ; 
}
inline unsigned int GNode::getIndex()
{
    return m_index ; 
}
inline void GNode::setParent(GNode* parent)
{ 
    m_parent = parent ; 
}
inline GNode* GNode::getParent()
{
    return m_parent ; 
}
inline char* GNode::getDescription()
{
    return m_description ;
}
inline void GNode::setDescription(char* description)
{ 
    m_description = strdup(description) ; 
}
inline void GNode::addChild(GNode* child)
{
    m_children.push_back(child);
}
inline GNode* GNode::getChild(unsigned int index)
{
    return index < getNumChildren() ? m_children[index] : NULL ;
}
inline unsigned int GNode::getNumChildren()
{
    return m_children.size();
}

inline void GNode::setLevelTransform(GMatrixF* ltransform)
{
   m_ltransform = ltransform ; 
}
inline GMatrixF* GNode::getLevelTransform()
{
   return m_ltransform ; 
}

inline void GNode::setName(const char* name)
{
    m_name = strdup(name); 
}
inline const char* GNode::getName()
{
    return m_name ; 
}
inline void GNode::setRepeatIndex(unsigned int index)
{
    m_repeat_index = index ; 
}
inline unsigned int GNode::getRepeatIndex()
{
    return m_repeat_index ; 
}

