#pragma once

#include <vector>
#include <map>
#include <glm/glm.hpp>

class GMesh ; 

struct BBox {
   glm::vec3 min ; 
   glm::vec3 max ; 
};


template <typename MeshT>
class MWrap { 
    // Developed with single/few mesh caches in mind like --jdyb --kdyb 
   public:
       MWrap(MeshT* mesh);
       MeshT* getMesh();
       unsigned int getNumVertices();
       unsigned int getNumFaces();
   public:
       GMesh* createGMesh();   // must externally setName and setIndex for usage with GGeo
       void load(GMesh* mm);   // GMergedMesh isa GMesh
       void copyIn( float* vdata, unsigned int num_vertices, unsigned int* fdata, unsigned int num_faces );
       void copyOut(float* vdata, unsigned int num_vertices, unsigned int* fdata, unsigned int num_faces, float* ndata );
       void createWithWeldedBoundary(MWrap<MeshT>* wa, MWrap<MeshT>* wb, std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle>& a2b);
   public:
      // recursive traverse over vertices labelling separately connected component vertices with an index
      int labelConnectedComponentVertices(const char* vpropname);
      void calcFaceCentroids(const char* fpropname);
   public:
       std::vector<typename MeshT::VertexHandle>& getBoundaryLoop();
   public:
      // local state changing
      unsigned int collectBoundaryLoop();
      void findBounds();
   public:
      // mesh changing 
      unsigned int deleteFaces(const char* fpredicate_name );
   public:
      void dump(const char* msg="MWrap::dump", unsigned int detail=1);
      void dumpBounds(const char* msg="MWrap::dumpBounds");
      void dumpStats(const char* msg="MWrap::dumpStats");
      void dumpFaces(const char* msg="MWrap::dumpFaces", unsigned int detail=1);
      void dumpVertices(const char* msg="MWrap::dumpVertices", unsigned int detail=1);
      void dumpBoundaryLoop(const char* msg="MWrap::dumpBoundaryLoop");
   public:
      void copyTo(MeshT* dst, std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle>& src2dst );
      void partialCopyTo(MeshT* dst, const char* ivpropname, int ivpropval, std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle>& src2dst );

   public:
      static int labelSpatialPairs(MeshT* a, MeshT* b, glm::vec4 delta, const char* fposprop, const char* fpropname);
      static std::map<typename MeshT::VertexHandle, typename MeshT::VertexHandle> findBoundaryVertexMap(MWrap<MeshT>* wa, MWrap<MeshT>* wb);
   public:
      static MWrap<MeshT>* load(const char* dir);
      void save(const char* dir, const char* version="_v0");
      void write(const char* tmpl, unsigned int index);

   private:
       MeshT* m_mesh ; 
   private:
       BBox   m_bbox ; 
       std::vector<typename MeshT::VertexHandle>  m_boundary ; 


};


template <typename MeshT>
inline MWrap<MeshT>::MWrap(MeshT* mesh) : m_mesh(mesh) 
{
}

template <typename MeshT>
inline MeshT* MWrap<MeshT>::getMesh()
{
    return m_mesh ; 
}

template <typename MeshT>
inline std::vector<typename MeshT::VertexHandle>& MWrap<MeshT>::getBoundaryLoop()
{
    return m_boundary ; 
}


