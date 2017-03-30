#pragma once

#include <functional>
#include <vector>

#include <glm/glm.hpp>

#include "NQuad.hpp"
#include "NBBox.hpp"

class Timer ; 
struct NFieldGrid3 ; 
struct OctreeDrawInfo ;

class OctreeNode
{
public:

    enum OctreeNodeType
    {
        Node_None,
        Node_Internal,
        Node_Psuedo,
        Node_Leaf,
    };

    template <typename T>
    static int Corners( const T& arg_min, NFieldGrid3* f, const nvec4& ce, const int ncorner=8, const int size=1 );

    static void PopulateLeaf(int corners, OctreeNode* leaf, NFieldGrid3* f, const nvec4& ce );
    static void DestroyOctree(OctreeNode* node) ;

    static void GenerateVertexIndices(OctreeNode* node, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, const nbbox& bb, const nvec4& ce, NFieldGrid3* fg);

    static void ContourCellProc(OctreeNode* node, std::vector<int>& indexBuffer);

    static OctreeNode* ConstructOctreeNodes(OctreeNode* node, NFieldGrid3* fg, const nvec4& ce, int& count);

    static OctreeNode* SimplifyOctree(OctreeNode* node, float threshold);


	OctreeNode()
		: type(Node_None)
		, min(0, 0, 0)
		, size(0)
		, drawInfo(nullptr)
	{
		for (int i = 0; i < 8; i++)
		{
			children[i] = nullptr;
		}
	}


	OctreeNodeType	type;
	glm::ivec3		min;
	int				size;
	OctreeNode*		children[8];
	OctreeDrawInfo*	drawInfo;
};




inline bool operator == ( const OctreeNode& a, const OctreeNode& b)
{
    return a.type == b.type && 
           a.size == b.size &&
           a.min == b.min  ;
}



