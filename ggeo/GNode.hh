#pragma once

#include <string>
#include <vector>

#include <glm/fwd.hpp>
#include "GVector.hh"

template <typename T> class GMatrix ; 
class GMesh ;
class GSolid ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GNode {
  public:
      GNode(unsigned int index, GMatrix<float>* transform, const GMesh* mesh);
      void setIndex(unsigned int index);
      void setSelected(bool selected);
      bool isSelected();
      virtual ~GNode();
  private:
      void init();

  public: 
      void Summary(const char* msg="GNode::Summary");
      void dump(const char* msg="GNode::dump");
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
      GNode*        getParent() const ; 
      GNode*        getChild(unsigned index);
      GSolid*       getChildSolid(unsigned index);
      unsigned int  getNumChildren();
      char*         getDescription();
      gfloat3*      getLow();
      gfloat3*      getHigh();
      const GMesh*  getMesh();
      unsigned      getMeshIndex() const ;

  public:
      unsigned int* getNodeIndices();
      unsigned int* getBoundaryIndices();
      unsigned int* getSensorIndices();

  public:
     void updateBounds();
     void updateBounds(gfloat3& low, gfloat3& high );

  public:
      glm::mat4 getTransformMat4();


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
      unsigned int         getLastProgenyCount();
      unsigned int         getProgenyNumVertices();  // includes self when m_selfdigest is true
      GNode*               findProgenyDigest(const std::string& pdig) ;
      std::vector<GNode*>  findAllProgenyDigest(std::string& dig);
      std::vector<GNode*>  findAllInstances(unsigned ridx, bool inside, bool honour_selection );
  private:
      std::string          meshDigest();
      std::string          localDigest();
      static std::string   localDigest(std::vector<GNode*>& nodes, GNode* extra=NULL);

      void collectProgeny(std::vector<GNode*>& progeny);
      void collectAllProgenyDigest(std::vector<GNode*>& match, std::string& dig);
      void collectAllInstances(std::vector<GNode*>& match, unsigned ridx, bool inside, bool honour_selection );


  private:
      bool                m_selfdigest ; // when true getProgenyDigest includes self node 
      bool                m_selected ;
  protected: 
      unsigned int        m_index ; 
  private:
      GNode*              m_parent ; 
      std::vector<GNode*> m_children ;
      char*               m_description ;

  private: 
      GMatrixF*           m_transform ; 
      GMatrixF*           m_ltransform ; 
  protected: 
      const GMesh*        m_mesh ; 
  private: 
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


#include "GGEO_TAIL.hh"


