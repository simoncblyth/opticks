#pragma once

#include <functional>
#include <vector>

#include <glm/glm.hpp>

#include "NQuad.hpp"
#include "NBBox.hpp"

class Timer ; 
template <typename FVec, typename IVec> struct NFieldGrid3 ; 

typedef NFieldGrid3<glm::vec3,glm::ivec3> FG3 ; 


struct OctreeDrawInfo ;
struct FGLite ; 


class OctreeNode ; 


class OctreeMgr
{
    public:
        OctreeMgr(OctreeNode* root, float threshold) 
           : 
            m_root(root), 
            m_threshold(threshold),
            m_node_count(0), 
            m_qef_nan(0), 
            m_qef_oob(0) 
          {};

        OctreeNode* simplify();

    private:
        OctreeNode* simplify_r(OctreeNode* node, int depth);
    private:
        OctreeNode* m_root ; 
        float       m_threshold ; 
        int         m_node_count ; 
        int         m_qef_nan ; 
        int         m_qef_oob ; 
};



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

    static int Corners( const glm::ivec3& min, FG3* fg, const int ncorner=8, const int size=1 );

    static OctreeNode* MakeLeaf(const glm::ivec3& min,  int corners, FG3* fg, int size );

    static void PopulateLeaf(int corners, OctreeNode* leaf, FG3* f);
    static void DestroyOctree(OctreeNode* node) ;

    static void GenerateVertexIndices(OctreeNode* node, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, FG3* fg, FGLite* fgl);

    static void ContourCellProc(OctreeNode* node, std::vector<int>& indexBuffer);

    static OctreeNode* ConstructOctreeNodes(OctreeNode* node, FG3* fg, int& count);

    static OctreeNode* SimplifyOctree(OctreeNode* node, float threshold);


	OctreeNode()
		: type(Node_None)
		, min(0, 0, 0)
		, size(0)
		, drawInfo(NULL)
	{
		for (int i = 0; i < 8; i++) children[i] = NULL ; 
	}


	OctreeNodeType	type;
	glm::ivec3		min;
	int				size;
	OctreeNode*		children[8];
	OctreeDrawInfo*	drawInfo;
};


struct OctCheck 
{
    OctCheck(OctreeNode* root) 
       : 
       node_count(0),
       bad_node(0),
       maxdepth(0)
       {
           Check(root, 0);
       } ; 

    void Check(OctreeNode* node, int depth=0 );

    bool ok(){ return bad_node == 0 ; }

    void report(const char* msg);


    int node_count ; 
    int bad_node   ; 
    int maxdepth   ; 
};




inline bool operator == ( const OctreeNode& a, const OctreeNode& b)
{
    return a.type == b.type && 
           a.size == b.size &&
           a.min == b.min  ;
}



