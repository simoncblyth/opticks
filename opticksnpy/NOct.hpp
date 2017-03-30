#pragma once
#include <vector>

#include "NGLM.hpp"
#include "NQuad.hpp"
#include "NBBox.hpp"

#include "NPY_API_EXPORT.hh"

struct NFieldGrid3 ; 


class NPY_API NOct
{
public:
    enum 
    {
        Node_None,
        Node_Internal,
        Node_Psuedo,
        Node_Leaf,
    };

    template <typename T>
    static int Corners( const T& ijk, NFieldGrid3* f, const nvec4& ce, const int ncorner=8, const int size=1 );

    static void PopulateLeaf(int corners, NOct* leaf, NFieldGrid3* f, const nvec4& ce );

    static void GenerateVertexIndices(NOct* node, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, const nbbox& bb, const nvec4& ce, NFieldGrid3* fg);
    static void ContourCellProc(NOct* node, std::vector<int>& indexBuffer);
    static NOct* ConstructOctreeNodes(NOct* node, NFieldGrid3* fg, const nvec4& ce, int& count);
    static NOct* SimplifyOctree(NOct* node, float threshold);



    NOct()
        : 
          type(Node_None),
          min(0, 0, 0),
          size(0)
    {
          for (int i = 0; i < 8; i++) children[i] = NULL ; 
    }

    int             type;
    glm::ivec3      min;
    int             size;
    NOct*           children[8];

};



inline bool operator == ( const NOct& a, const NOct& b)
{
    return a.type == b.type && 
           a.size == b.size &&
           a.min == b.min  ;
}




