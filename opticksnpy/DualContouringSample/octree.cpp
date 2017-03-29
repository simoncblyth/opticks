/*

Implementations of Octree member functions.

Copyright (C) 2011  Tao Ju

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License
(LGPL) as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "octree.h"
#include <iostream>
#include <iomanip>
#include <bitset>



#include "Timer.hpp"
#include "GLMFormat.hpp"
#include "PLOG.hh"
#include  <boost/unordered_map.hpp>


#include "NTreeTools.hpp"
template class NTraverser<OctreeNode,8> ; 
template class NComparer<OctreeNode,8> ; 


#include "NBBox.hpp"
#include "NGrid3.hpp"
#include "NField3.hpp"


float Density_Func(std::function<float(float,float,float)>* f, const nvec4& ce, const vec3& ijk)
{
    nvec3 p ; 
    p.x = ce.x + ijk.x*ce.w ; 
    p.y = ce.y + ijk.y*ce.w ; 
    p.z = ce.z + ijk.z*ce.w ; 

    float fp = (*f)(p.x, p.y, p.z);
    return fp ; 
}


// ----------------------------------------------------------------------------

const int MATERIAL_AIR = 0;
const int MATERIAL_SOLID = 1;

const float QEF_ERROR = 1e-6f;
const int QEF_SWEEPS = 4;

// ----------------------------------------------------------------------------

const ivec3 CHILD_MIN_OFFSETS[] =
{
	// needs to match the vertMap from Dual Contouring impl
	ivec3( 0, 0, 0 ),
	ivec3( 0, 0, 1 ),
	ivec3( 0, 1, 0 ),
	ivec3( 0, 1, 1 ),
	ivec3( 1, 0, 0 ),
	ivec3( 1, 0, 1 ),
	ivec3( 1, 1, 0 ),
	ivec3( 1, 1, 1 ),
};


// ----------------------------------------------------------------------------
// data from the original DC impl, drives the contouring process

const int edgevmap[12][2] = 
{
	{0,4},{1,5},{2,6},{3,7},	// x-axis 
	{0,2},{1,3},{4,6},{5,7},	// y-axis
	{0,1},{2,3},{4,5},{6,7}		// z-axis
};


const int cellProcFaceMask[12][3] = {{0,4,0},{1,5,0},{2,6,0},{3,7,0},{0,2,1},{4,6,1},{1,3,1},{5,7,1},{0,1,2},{2,3,2},{4,5,2},{6,7,2}} ;
const int cellProcEdgeMask[6][5] = {{0,1,2,3,0},{4,5,6,7,0},{0,4,1,5,1},{2,6,3,7,1},{0,2,4,6,2},{1,3,5,7,2}} ;

const int faceProcFaceMask[3][4][3] = {
	{{4,0,0},{5,1,0},{6,2,0},{7,3,0}},
	{{2,0,1},{6,4,1},{3,1,1},{7,5,1}},
	{{1,0,2},{3,2,2},{5,4,2},{7,6,2}}
} ;

const int faceProcEdgeMask[3][4][6] = {
	{{1,4,0,5,1,1},{1,6,2,7,3,1},{0,4,6,0,2,2},{0,5,7,1,3,2}},
	{{0,2,3,0,1,0},{0,6,7,4,5,0},{1,2,0,6,4,2},{1,3,1,7,5,2}},
	{{1,1,0,3,2,0},{1,5,4,7,6,0},{0,1,5,0,4,1},{0,3,7,2,6,1}}
};

const int edgeProcEdgeMask[3][2][5] = {
	{{3,2,1,0,0},{7,6,5,4,0}},
	{{5,1,4,0,1},{7,3,6,2,1}},
	{{6,4,2,0,2},{7,5,3,1,2}},
};

const int processEdgeMask[3][4] = {{3,2,1,0},{7,5,6,4},{11,10,9,8}} ;

// -------------------------------------------------------------------------------

OctreeNode* SimplifyOctree(OctreeNode* node, float threshold)
{
	if (!node)
	{
		return NULL;
	}

	if (node->type != Node_Internal)
	{
		// can't simplify!
		return node;
	}

	svd::QefSolver qef;
	int signs[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	int midsign = -1;
	int edgeCount = 0;
	bool isCollapsible = true;

	for (int i = 0; i < 8; i++)
	{
		node->children[i] = SimplifyOctree(node->children[i], threshold);
		if (node->children[i])
		{
			OctreeNode* child = node->children[i];
			if (child->type == Node_Internal)
			{
				isCollapsible = false;
			}
			else
			{
				qef.add(child->drawInfo->qef);

				midsign = (child->drawInfo->corners >> (7 - i)) & 1; 
				signs[i] = (child->drawInfo->corners >> i) & 1; 

				edgeCount++;
			}
		}
	}

	if (!isCollapsible)
	{
		// at least one child is an internal node, can't collapse
		return node;
	}

	svd::Vec3 qefPosition;
	qef.solve(qefPosition, QEF_ERROR, QEF_SWEEPS, QEF_ERROR);
	float error = qef.getError();

	// convert to glm vec3 for ease of use
	vec3 position(qefPosition.x, qefPosition.y, qefPosition.z);

	// at this point the masspoint will actually be a sum, so divide to make it the average
	if (error > threshold)
	{
		// this collapse breaches the threshold
		return node;
	}

	if (position.x < node->min.x || position.x > (node->min.x + node->size) ||
		position.y < node->min.y || position.y > (node->min.y + node->size) ||
		position.z < node->min.z || position.z > (node->min.z + node->size))
	{
		const auto& mp = qef.getMassPoint();
		position = vec3(mp.x, mp.y, mp.z);
	}

	// change the node from an internal node to a 'psuedo leaf' node
	OctreeDrawInfo* drawInfo = new OctreeDrawInfo;

	for (int i = 0; i < 8; i++)
	{
		if (signs[i] == -1)
		{
			// Undetermined, use centre sign instead
			drawInfo->corners |= (midsign << i);
		}
		else 
		{
			drawInfo->corners |= (signs[i] << i);
		}
	}

	drawInfo->averageNormal = vec3(0.f);
	for (int i = 0; i < 8; i++)
	{
		if (node->children[i])
		{
			OctreeNode* child = node->children[i];
			if (child->type == Node_Psuedo || 
				child->type == Node_Leaf)
			{
				drawInfo->averageNormal += child->drawInfo->averageNormal;
			}
		}
	}

	drawInfo->averageNormal = glm::normalize(drawInfo->averageNormal);
	drawInfo->position = position;
	drawInfo->qef = qef.getData();

	for (int i = 0; i < 8; i++)
	{
		DestroyOctree(node->children[i]);
		node->children[i] = nullptr;
	}

	node->type = Node_Psuedo;
	node->drawInfo = drawInfo;

	return node;
}

// ----------------------------------------------------------------------------

void GenerateVertexIndices(OctreeNode* node, VertexBuffer& vertexBuffer, const nbbox& bb, const nvec4& ce)
{
	if (!node)
	{
		return;
	}

	if (node->type != Node_Leaf)
	{
		for (int i = 0; i < 8; i++)
		{
			GenerateVertexIndices(node->children[i], vertexBuffer, bb, ce);
		}
	}

	if (node->type != Node_Internal)
	{
		OctreeDrawInfo* d = node->drawInfo;
		if (!d)
		{
			printf("Error! Could not add vertex!\n");
			exit(EXIT_FAILURE);
		}

		d->index = vertexBuffer.size();

        vec3 ijk = d->position ; 

      /*
        vec3 xyz ; 
        xyz.x = bb.min.x + ijk.x*bb.side.x ; 
        xyz.y = bb.min.y + ijk.y*bb.side.y ; 
        xyz.z = bb.min.z + ijk.z*bb.side.z ; 
      */

        vec3 xyz ;
        xyz.x = ce.x + ijk.x*ce.w ; 
        xyz.y = ce.y + ijk.y*ce.w ; 
        xyz.z = ce.z + ijk.z*ce.w ; 


		vertexBuffer.push_back(MeshVertex(xyz, d->averageNormal));
	}
}

// ----------------------------------------------------------------------------

void ContourProcessEdge(OctreeNode* node[4], int dir, IndexBuffer& indexBuffer)
{
	int minSize = 1000000;		// arbitrary big number
	int minIndex = 0;
	int indices[4] = { -1, -1, -1, -1 };
	bool flip = false;
	bool signChange[4] = { false, false, false, false };

	for (int i = 0; i < 4; i++)
	{
		const int edge = processEdgeMask[dir][i];
		const int c1 = edgevmap[edge][0];
		const int c2 = edgevmap[edge][1];

		const int m1 = (node[i]->drawInfo->corners >> c1) & 1;
		const int m2 = (node[i]->drawInfo->corners >> c2) & 1;

		if (node[i]->size < minSize)
		{
			minSize = node[i]->size;
			minIndex = i;
			flip = m1 != MATERIAL_AIR; 
		}

		indices[i] = node[i]->drawInfo->index;

		signChange[i] = 
			(m1 == MATERIAL_AIR && m2 != MATERIAL_AIR) ||
			(m1 != MATERIAL_AIR && m2 == MATERIAL_AIR);
	}

	if (signChange[minIndex])
	{
		if (!flip)
		{
			indexBuffer.push_back(indices[0]);
			indexBuffer.push_back(indices[1]);
			indexBuffer.push_back(indices[3]);

			indexBuffer.push_back(indices[0]);
			indexBuffer.push_back(indices[3]);
			indexBuffer.push_back(indices[2]);
		}
		else
		{
			indexBuffer.push_back(indices[0]);
			indexBuffer.push_back(indices[3]);
			indexBuffer.push_back(indices[1]);

			indexBuffer.push_back(indices[0]);
			indexBuffer.push_back(indices[2]);
			indexBuffer.push_back(indices[3]);
		}
	}
}

// ----------------------------------------------------------------------------

void ContourEdgeProc(OctreeNode* node[4], int dir, IndexBuffer& indexBuffer)
{
	if (!node[0] || !node[1] || !node[2] || !node[3])
	{
		return;
	}

	if (node[0]->type != Node_Internal &&
		node[1]->type != Node_Internal &&
		node[2]->type != Node_Internal &&
		node[3]->type != Node_Internal)
	{
		ContourProcessEdge(node, dir, indexBuffer);
	}
	else
	{
		for (int i = 0; i < 2; i++)
		{
			OctreeNode* edgeNodes[4];
			const int c[4] = 
			{
				edgeProcEdgeMask[dir][i][0],
				edgeProcEdgeMask[dir][i][1],
				edgeProcEdgeMask[dir][i][2],
				edgeProcEdgeMask[dir][i][3],
			};

			for (int j = 0; j < 4; j++)
			{
				if (node[j]->type == Node_Leaf || node[j]->type == Node_Psuedo)
				{
					edgeNodes[j] = node[j];
				}
				else
				{
					edgeNodes[j] = node[j]->children[c[j]];
				}
			}

			ContourEdgeProc(edgeNodes, edgeProcEdgeMask[dir][i][4], indexBuffer);
		}
	}
}

// ----------------------------------------------------------------------------

void ContourFaceProc(OctreeNode* node[2], int dir, IndexBuffer& indexBuffer)
{
	if (!node[0] || !node[1])
	{
		return;
	}

	if (node[0]->type == Node_Internal || 
		node[1]->type == Node_Internal)
	{
		for (int i = 0; i < 4; i++)
		{
			OctreeNode* faceNodes[2];
			const int c[2] = 
			{
				faceProcFaceMask[dir][i][0], 
				faceProcFaceMask[dir][i][1], 
			};

			for (int j = 0; j < 2; j++)
			{
				if (node[j]->type != Node_Internal)
				{
					faceNodes[j] = node[j];
				}
				else
				{
					faceNodes[j] = node[j]->children[c[j]];
				}
			}

			ContourFaceProc(faceNodes, faceProcFaceMask[dir][i][2], indexBuffer);
		}
		
		const int orders[2][4] =
		{
			{ 0, 0, 1, 1 },
			{ 0, 1, 0, 1 },
		};
		for (int i = 0; i < 4; i++)
		{
			OctreeNode* edgeNodes[4];
			const int c[4] =
			{
				faceProcEdgeMask[dir][i][1],
				faceProcEdgeMask[dir][i][2],
				faceProcEdgeMask[dir][i][3],
				faceProcEdgeMask[dir][i][4],
			};

			const int* order = orders[faceProcEdgeMask[dir][i][0]];
			for (int j = 0; j < 4; j++)
			{
				if (node[order[j]]->type == Node_Leaf ||
					node[order[j]]->type == Node_Psuedo)
				{
					edgeNodes[j] = node[order[j]];
				}
				else
				{
					edgeNodes[j] = node[order[j]]->children[c[j]];
				}
			}

			ContourEdgeProc(edgeNodes, faceProcEdgeMask[dir][i][5], indexBuffer);
		}
	}
}

// ----------------------------------------------------------------------------

void ContourCellProc(OctreeNode* node, IndexBuffer& indexBuffer)
{
	if (node == NULL)
	{
		return;
	}

	if (node->type == Node_Internal)
	{
		for (int i = 0; i < 8; i++)
		{
			ContourCellProc(node->children[i], indexBuffer);
		}

		for (int i = 0; i < 12; i++)
		{
			OctreeNode* faceNodes[2];
			const int c[2] = { cellProcFaceMask[i][0], cellProcFaceMask[i][1] };
			
			faceNodes[0] = node->children[c[0]];
			faceNodes[1] = node->children[c[1]];

			ContourFaceProc(faceNodes, cellProcFaceMask[i][2], indexBuffer);
		}

		for (int i = 0; i < 6; i++)
		{
			OctreeNode* edgeNodes[4];
			const int c[4] = 
			{
				cellProcEdgeMask[i][0],
				cellProcEdgeMask[i][1],
				cellProcEdgeMask[i][2],
				cellProcEdgeMask[i][3],
			};

			for (int j = 0; j < 4; j++)
			{
				edgeNodes[j] = node->children[c[j]];
			}

			ContourEdgeProc(edgeNodes, cellProcEdgeMask[i][4], indexBuffer);
		}
	}
}


vec3 ApproximateZeroCrossingPosition(const vec3& p0, const vec3& p1, std::function<float(float,float,float)>* f, const nvec4& ce)
{
	// approximate the zero crossing by finding the min value along the edge
	float minValue = 100000.f;
	float t = 0.f;
	float currentT = 0.f;
	const int steps = 8;
	const float increment = 1.f / (float)steps;
	while (currentT <= 1.f)
	{
		const vec3 p = p0 + ((p1 - p0) * currentT);
		const float density = glm::abs(Density_Func(f,ce,p));
		if (density < minValue)
		{
			minValue = density;
			t = currentT;
		}

		currentT += increment;
	}

	return p0 + ((p1 - p0) * t);
}

// ----------------------------------------------------------------------------

vec3 CalculateSurfaceNormal(const vec3& p, std::function<float(float,float,float)>* f, const nvec4& ce)
{
	const float H = 0.001f; // hmm delta in ijk-space converted to floats 
	const float dx = Density_Func(f,ce,p + vec3(H, 0.f, 0.f)) - Density_Func(f,ce,p - vec3(H, 0.f, 0.f));
	const float dy = Density_Func(f,ce,p + vec3(0.f, H, 0.f)) - Density_Func(f,ce,p - vec3(0.f, H, 0.f));
	const float dz = Density_Func(f,ce,p + vec3(0.f, 0.f, H)) - Density_Func(f,ce,p - vec3(0.f, 0.f, H));

	return glm::normalize(vec3(dx, dy, dz));
}



template <typename T>
int Corners( const T& arg_min, std::function<float(float,float,float)>* f, const nvec4& ce, const int ncorner=8, const int size=1 )
{
    const ivec3 leaf_min ; 
    leaf_min.x = arg_min.x ; 
    leaf_min.y = arg_min.y ; 
    leaf_min.z = arg_min.z ; 

	int corners = 0;
	for (int i = 0; i < ncorner; i++)
	{
		const ivec3 cornerPos = leaf_min + size*CHILD_MIN_OFFSETS[i];
		const float density = Density_Func(f, ce, vec3(cornerPos));
		const int material = density < 0.f ? MATERIAL_SOLID : MATERIAL_AIR;
		corners |= (material << i);
	}
    return corners ; 
}


void PopulateLeaf(int corners, OctreeNode* leaf, std::function<float(float,float,float)>* f, const nvec4& ce )
{
	// otherwise the voxel contains the surface, so find the edge intersections
	const int MAX_CROSSINGS = 6;
	int edgeCount = 0;
	vec3 averageNormal(0.f);
	svd::QefSolver qef;

	for (int i = 0; i < 12 && edgeCount < MAX_CROSSINGS; i++)
	{
		const int c1 = edgevmap[i][0];
		const int c2 = edgevmap[i][1];

		const int m1 = (corners >> c1) & 1;
		const int m2 = (corners >> c2) & 1;

		if ((m1 == MATERIAL_AIR && m2 == MATERIAL_AIR) ||
			(m1 == MATERIAL_SOLID && m2 == MATERIAL_SOLID))
		{
			// no zero crossing on this edge
			continue;
		}

		const vec3 p1 = vec3(leaf->min + CHILD_MIN_OFFSETS[c1]);
		const vec3 p2 = vec3(leaf->min + CHILD_MIN_OFFSETS[c2]);
		const vec3 p = ApproximateZeroCrossingPosition(p1, p2, f, ce);
		const vec3 n = CalculateSurfaceNormal(p, f, ce);
		qef.add(p.x, p.y, p.z, n.x, n.y, n.z);

		averageNormal += n;

		edgeCount++;
	}

	svd::Vec3 qefPosition;
	qef.solve(qefPosition, QEF_ERROR, QEF_SWEEPS, QEF_ERROR);

	OctreeDrawInfo* drawInfo = new OctreeDrawInfo;
	drawInfo->position = vec3(qefPosition.x, qefPosition.y, qefPosition.z);
	drawInfo->qef = qef.getData();

	const vec3 min = vec3(leaf->min);
	const vec3 max = vec3(leaf->min + ivec3(leaf->size));
	if (drawInfo->position.x < min.x || drawInfo->position.x > max.x ||
		drawInfo->position.y < min.y || drawInfo->position.y > max.y ||
		drawInfo->position.z < min.z || drawInfo->position.z > max.z)
	{
		const auto& mp = qef.getMassPoint();
		drawInfo->position = vec3(mp.x, mp.y, mp.z);
	}

	drawInfo->averageNormal = glm::normalize(averageNormal / (float)edgeCount);
	drawInfo->corners = corners;

	leaf->type = Node_Leaf;
	leaf->drawInfo = drawInfo;

}


OctreeNode* ConstructLeaf(OctreeNode* leaf, std::function<float(float,float,float)>* f, const nvec4& ce )
{
    assert(leaf && leaf->size == 1);

	int corners = Corners( leaf->min, f, ce);

	if (corners == 0 || corners == 255)
	{
		// voxel is full inside or outside the volume
		delete leaf;
		return nullptr;
	}

   PopulateLeaf( corners, leaf, f, ce ) ;

   return leaf ;
}


int PopulateLeaves( OctreeNode* node, const int childSize,  std::function<float(float,float,float)>* f, const nvec4& ce)
{
    int nleaf = 0 ; 
    for (int i = 0; i < 8; i++)
    {
        ivec3 leaf_min = node->min + (CHILD_MIN_OFFSETS[i] * childSize); 
        int corners = Corners(leaf_min, f, ce );  
        if(corners == 0 || corners == 255) continue ; 

        nleaf++ ; 
        OctreeNode* leaf = new OctreeNode;
        leaf->size = childSize;
        leaf->min = leaf_min ; 

        PopulateLeaf( corners, leaf, f, ce ) ; 

        node->children[i] = leaf ;
    } 
    return nleaf ; 
}


OctreeNode* ConstructOctreeNodes(OctreeNode* node, std::function<float(float,float,float)>* f, const nvec4& ce, int& count)
{
    assert(node && node->size > 1);
	bool hasChildren = false;

    // count++ ;  // ConstructOctreeNodes recursive invokations

	const int childSize = node->size / 2;
    if(childSize == 1)
    {
        int nleaf = PopulateLeaves(node, childSize, f, ce);
        hasChildren = nleaf > 0 ; 
        count += 8 ; // candidate leaves tested   
    } 
    else
    {
        int nchild = 0 ;  
        for (int i = 0; i < 8; i++)
        {
            ivec3 child_min = node->min + (CHILD_MIN_OFFSETS[i] * childSize) ;

            OctreeNode* child = new OctreeNode;
            child->size = childSize;
            child->min = child_min ; 
            child->type = Node_Internal;

            OctreeNode* confirmed = ConstructOctreeNodes(child, f, ce, count);
            if(confirmed) nchild++ ; 

            node->children[i] = confirmed ; 
        }
        hasChildren = nchild > 0 ; 
    }

	if (!hasChildren)
	{
		delete node;
		return nullptr;
	}
	return node;
}





class Constructor 
{
    static const int maxlevel = 10 ; 

    typedef std::function<float(float,float,float)> F ; 
    typedef boost::unordered_map<unsigned, OctreeNode*> UMAP ;
    UMAP cache[maxlevel] ; 

    public:
        Constructor(F* f, const nvec4& ce, const nbbox& bb, int nominal, int coarse );
        OctreeNode* create();
        void dump();
        void scan(const char* msg="scan", int depth=2, int limit=30 ) const ; 

        void dump_domain(const char* msg="dump_domain") const ;
        nvec3 position(const ivec3& offset_ijk) const ;
        nvec3 position_bb(const ivec3& offset_ijk, int depth) const ;
        float density(const ivec3& offset_ijk) const ;
        float density_bb(const ivec3& offset_ijk, int depth) const ;


    private:
        OctreeNode* create_coarse_nominal();
        OctreeNode* create_nominal();
        void buildBottomUpFromLeaf(int leaf_loc, OctreeNode* leaf );
    private:
        NMultiGrid3 m_mgrid ; 

        F*          m_func ; 
        nvec4       m_ce ;  
        nbbox       m_bb ; 

        NGrid3*     m_nominal ; 
        NGrid3*     m_coarse ; 
        NGrid3*     m_subtile ; 
        NGrid3*     m_dgrid ; 

        nivec3      m_nominal_min ; 
        int         m_upscale_factor ; 

        OctreeNode* m_root ; 

        unsigned m_num_leaf ; 
        unsigned m_num_from_cache ; 
        unsigned m_num_into_cache ; 
};


nvec3 Constructor::position(const ivec3& offset_ijk) const 
{
    // hmm the jiggery poked extent contains the ijk extent, so this only valid at nominal
    nvec3 xyz ; 
    xyz.x = m_ce.x + offset_ijk.x*m_ce.w ; 
    xyz.y = m_ce.y + offset_ijk.y*m_ce.w ; 
    xyz.z = m_ce.z + offset_ijk.z*m_ce.w ; 
    return xyz ; 
}


nvec3 Constructor::position_bb(const ivec3& natural_ijk, int depth) const 
{
    assert( depth >= 0 && depth <= m_nominal->level  ); 
    NGrid3* dgrid = m_mgrid.grid[depth] ; 
     
    assert( natural_ijk.x >= 0 && natural_ijk.x <= dgrid->size );
    assert( natural_ijk.y >= 0 && natural_ijk.y <= dgrid->size );
    assert( natural_ijk.z >= 0 && natural_ijk.z <= dgrid->size );

    /*
    const nivec3& half_min = dgrid->half_min ; 
    nivec3 ijk = {0,0,0} ;
    ijk.x = offset_ijk.x - half_min.x ; 
    ijk.y = offset_ijk.y - half_min.y ; 
    ijk.z = offset_ijk.z - half_min.z ; 
    */


    nivec3 _natural_ijk( natural_ijk.x, natural_ijk.y, natural_ijk.z );
    nvec3 frac_pos = dgrid->fpos<nivec3>(_natural_ijk);

    nvec3 world_pos ; 
    world_pos.x = m_bb.min.x + frac_pos.x*m_bb.side.x ; 
    world_pos.y = m_bb.min.y + frac_pos.y*m_bb.side.y ; 
    world_pos.z = m_bb.min.z + frac_pos.z*m_bb.side.z ; 

/*
    std::cout
           << " _natural_ijk " << _natural_ijk.desc()
           << " frac_pos " << frac_pos.desc() 
           << " bb.side " << m_bb.side.desc()
           << " world_pos " << world_pos.desc()
           << std::endl ;
*/


    return world_pos ; 
}



float Constructor::density_bb(const ivec3& natural_ijk, int depth) const 
{
    nvec3 world_pos = position_bb(natural_ijk, depth);
    return (*m_func)(world_pos.x, world_pos.y, world_pos.z);
}



float Constructor::density(const ivec3& offset_ijk) const 
{
    return Density_Func( m_func, m_ce, offset_ijk );
}

Constructor::Constructor(F* f, const nvec4& ce, const nbbox& bb, int nominal, int coarse )
   :
   m_func(f),  
   m_ce(ce),
   m_bb(bb),
   m_nominal(m_mgrid.grid[nominal]),  
   m_coarse( m_mgrid.grid[coarse] ),  
   m_subtile(NULL),  
   m_dgrid(NULL),  
   m_nominal_min(-m_nominal->size/2, -m_nominal->size/2, -m_nominal->size/2),
   m_upscale_factor(0),
   m_root(NULL),
   m_num_leaf(0),
   m_num_from_cache(0),
   m_num_into_cache(0)
{
   assert( coarse <= nominal && nominal < maxlevel ); 
   m_subtile = m_mgrid.grid[nominal-coarse] ;  
   m_upscale_factor = m_nominal->upscale_factor( *m_coarse );

   std::cout << "Constructor"
              << " upscale_factor " << m_upscale_factor
              << std::endl 
              << " nominal " << m_nominal->desc()
              << std::endl  
              << " coarse  " << m_coarse->desc()
              << std::endl  
              << " subtile " << m_subtile->desc()
              << std::endl ; 

   dump_domain(); 
   for(int d=0 ; d <= nominal ; d++) scan("scan", d, 16); 
}

void Constructor::dump()
{
    std::cout << "ConstructOctreeBottomUp"
              << " num_leaf " << m_num_leaf 
              << " num_into_cache " << m_num_into_cache 
              << " num_from_cache " << m_num_from_cache 
              << " num_leaf/nominal.nloc " << float(m_num_leaf)/float(m_nominal->nloc) 
              << std::endl ;
}

void Constructor::buildBottomUpFromLeaf(int leaf_loc, OctreeNode* leaf)
{
    OctreeNode* node = leaf ; 
    OctreeNode* dnode = NULL ; 

    int depth = m_nominal->level  ; // start from nominal level, with the leaves 
    int dloc = leaf_loc ; 
    unsigned dsize = 1 ; 
    unsigned dchild = dloc & 7 ;    // lowest 3 bits, gives child index in immediate parent

    // at each turn : decrement depth, right-shift morton code to that of parent, left shift size doubling it 
    while(depth >= 1)
    {
        depth-- ; 
        dloc >>= 3 ;     
        dsize <<= 1 ;     
        m_dgrid = m_mgrid.grid[depth] ; 

        UMAP::const_iterator it = cache[depth].find(dloc);
        if(it == cache[depth].end())
        {
            m_num_into_cache++ ; 
            nivec3 d_ijk = m_dgrid->ijk(dloc); 
            d_ijk *= dsize ;      // scale coordinates to nominal 
            d_ijk += m_nominal_min ;      // add offset 

            dnode = new OctreeNode ; 
            dnode->size = dsize ; 
            dnode->min = ivec3(d_ijk.x, d_ijk.y, d_ijk.z) ; 
            dnode->type = Node_Internal;

            cache[depth].emplace(dloc, dnode)  ;

            if(m_num_into_cache < 10)
            std::cout << "into_cache " 
                      << " num_into_cache " << m_num_into_cache
                      << " dloc " << std::setw(6) << dloc
                      << " d_ijk " << d_ijk.desc()
                      << " m_nominal_min " << m_nominal_min.desc()
                      << " dsize " << dsize
                      << std::endl ; 
        }
        else
        {
            m_num_from_cache++ ; 
            dnode = it->second ;     
        }

        dnode->children[dchild] = node ;  
        node = dnode ; 
        dchild = dloc & 7 ;  // child index for next round
    }              // up the heirarchy from each leaf to root
}


void Constructor::dump_domain(const char* msg) const 
{
    LOG(info) <<  msg 
              << " nominal_min " << m_nominal_min.desc()
              << " ce  " << m_ce.desc()
              << " bb "  << m_bb.desc()
              ;
        
    for(int depth = 0 ; depth <= m_nominal->level ; depth++ )
    {
        int size = 1 << depth ; 
        NGrid3* dgrid = m_mgrid.grid[depth] ; 
        assert(size == dgrid->size);
 
        std::cout << " depth " << depth
                  << " size " << size
                  << " grid " << dgrid->desc()
                  << std::endl ; 

        ivec3 dmin(-size/2);   

        for (int i = 0; i < 8; i++)
        {
             const ivec3 ijk = CHILD_MIN_OFFSETS[i] * dgrid->size ; 
             nvec3 xyz_bb = position_bb(ijk, depth);  
             float sdf_bb = density_bb(ijk, depth) ;

             const ivec3 offset_ijk = dmin + ijk ; 
             nvec3 xyz = position(offset_ijk);
             float sdf = density(offset_ijk) ;

             std::cout << " i " << std::setw(3) << i 
                       << " offset_ijk " << std::setw(20) << gformat(offset_ijk)
                       << " position " << std::setw(20) << xyz.desc()
                       << " position_bb " << std::setw(20) << xyz_bb.desc()
                       << " sdf " << sdf
                       << " sdf_bb " << sdf_bb
                       << std::endl ; 
        }
    }
}


void Constructor::scan(const char* msg, int depth, int limit ) const 
{
    NGrid3* dgrid = m_mgrid.grid[depth] ; 
    std::cout << " dgrid   " << dgrid->desc()  << std::endl ; 
    std::cout << " nominal " << m_nominal->desc() << std::endl ; 

    int scale_to_nominal = 1 << (m_nominal->level - dgrid->level ) ; 
    int upscale_factor = m_nominal->upscale_factor( *dgrid ); 

    LOG(info) << msg 
              << " nominal level " << m_nominal->level
              << " dgrid level " << dgrid->level
              << " scale_to_nominal " << scale_to_nominal 
              << " upscale_factor " << upscale_factor
              << " limit " << limit 
              ;

    assert( scale_to_nominal == upscale_factor );
    for(int c=0 ; c < dgrid->nloc ; c++) 
    {
        nivec3 raw = dgrid->ijk( c );
        nivec3 scaled = raw ; 
        scaled *= scale_to_nominal ;
        nivec3 offset = scaled ;
        offset += m_nominal_min ;  

        ivec3 off( offset.x, offset.y, offset.z );
        nvec3 xyz = position(off);
        float sdf = density(off); 

        if( c < limit)
        std::cout << " c " << std::setw(6) << c 
                  << " raw " << std::setw(20) << raw.desc() 
                  << " scaled " << std::setw(20) << scaled.desc() 
                  << " offset " << std::setw(20) << offset.desc() 
                  << " xyz "  << std::setw(20) << xyz.desc()
                  << " sdf " << std::setw(20) <<  sdf
                  << std::endl 
                  ; 
    }
}

OctreeNode* Constructor::create()
{
    OctreeNode* root = NULL ; 
    if( m_coarse->level == m_nominal->level )
    {
        root = create_nominal() ;
    }
    else
    {
        root = create_coarse_nominal() ;
    }
    return root ; 
}


OctreeNode* Constructor::create_coarse_nominal()
{
    int leaf_size = 1 ; 
    for(int c=0 ; c < m_coarse->nloc ; c++) 
    {
        nivec3 c_ijk = m_coarse->ijk( c );
        c_ijk *= m_subtile->size ;    // scale coarse coordinates up to nominal 
        c_ijk += m_nominal_min ; 

        int corners = Corners( c_ijk , m_func, m_ce, 8, m_upscale_factor ); 
        if(corners == 0 || corners == 255) continue ;   
 
        for(int s=0 ; s < m_subtile->nloc ; s++)  // over nominal(at level) voxels in coarse tile
        {
            nivec3 s_ijk = m_subtile->ijk( s );
            s_ijk += c_ijk ; 
 
            int corners = Corners( s_ijk, m_func, m_ce, 8, leaf_size ); 
            if(corners == 0 || corners == 255) continue ;  

            m_num_leaf++ ; 

            OctreeNode* leaf = new OctreeNode;
            leaf->size = leaf_size ;
            leaf->min = ivec3(s_ijk.x, s_ijk.y, s_ijk.z) ; 

            PopulateLeaf( corners, leaf, m_func, m_ce ) ; 

            nivec3 a_ijk = s_ijk - m_nominal_min ;   // take out the offset, need 0:128 range

            int leaf_loc = m_nominal->loc( a_ijk );

            buildBottomUpFromLeaf(leaf_loc, leaf);

        }   // over nominal voxels within coarse tile
    }       // over coarse tiles

    UMAP::const_iterator it0 = cache[0].find(0);
    OctreeNode* root = it0 == cache[0].end() ? NULL : it0->second ; 
    assert(root);
    return root ; 
}

OctreeNode* Constructor::create_nominal()
{
    int leaf_size = 1 ; 
    for(int c=0 ; c < m_nominal->nloc ; c++) 
    {
        nivec3 ijk = m_nominal->ijk( c );
        nivec3 offset_ijk = ijk + m_nominal_min ; 

        int corners = Corners( offset_ijk , m_func, m_ce, 8, leaf_size ); 
        if(corners == 0 || corners == 255) continue ;   
 
        m_num_leaf++ ; 

        OctreeNode* leaf = new OctreeNode;
        leaf->size = leaf_size ;
        leaf->min = ivec3(offset_ijk.x, offset_ijk.y, offset_ijk.z) ; 

        PopulateLeaf( corners, leaf, m_func, m_ce ) ; 

        buildBottomUpFromLeaf( c, leaf);
    }   
    UMAP::const_iterator it0 = cache[0].find(0);
    OctreeNode* root = it0 == cache[0].end() ? NULL : it0->second ; 
    assert(root);
    return root ; 
}







Manager::Manager( const unsigned ctrl,  const int nominal, const int coarse, const float threshold, F* func, const nbbox& bb, Timer* timer )
    :
    m_ctrl(ctrl),
    m_nominal_size( 1 << nominal ),
    m_threshold(threshold), 
    m_func(func),
    m_bb(bb), 
    m_timer(timer),
    m_ctor(NULL),
    m_bottom_up(NULL),
    m_top_down(NULL),
    m_raw(NULL),
    m_simplified(NULL)
{
    nvec4     bbce = m_bb.center_extent();

    float ijkExtent = m_nominal_size/2 ;      // eg 64.f
    float xyzExtent = bbce.w  ;
    float ijk2xyz = xyzExtent/ijkExtent ;   // octree -> real world coordinates

    m_ce = make_nvec4(bbce.x, bbce.y, bbce.z, ijk2xyz );

    m_ctor = new Constructor(func, m_ce, m_bb, nominal, coarse );

    LOG(info) << "Manager::Maneger"
              << " xyzExtent " << xyzExtent
              << " ijkExtent " << ijkExtent
              << " bbce " << bbce.desc()
              << " ce " << m_ce.desc()
              ;
}

void Manager::buildOctree()
{

    if( m_ctrl & BUILD_BOTTOM_UP )
    {
        m_timer->stamp("_ConstructOctreeBottomUp");

        m_bottom_up = m_ctor->create();
        m_ctor->dump();

        assert(m_bottom_up);
        m_timer->stamp("ConstructOctreeBottomUp");
        NTraverser<OctreeNode,8>(m_bottom_up, "bottom_up", 1, 30 );
    }

    if( m_ctrl & BUILD_TOP_DOWN )
    {
  	    OctreeNode* root0 = new OctreeNode;
  	    root0->min = ivec3(-m_nominal_size/2);   // <--- cause of great pain
	    root0->size = m_nominal_size;
	    root0->type = Node_Internal;

        m_timer->stamp("_ConstructOctreeNodes");
        int count = 0 ; 
	    m_top_down = ConstructOctreeNodes(root0, m_func, m_ce, count);
        m_timer->stamp("ConstructOctreeNodes");
        std::cout << "ConstructOctreeNodes count " << count << std::endl ; 
        NTraverser<OctreeNode,8>(m_top_down, "top_down", 1, 30);
    }

    if( m_ctrl & BUILD_BOTH )
    {
        m_timer->stamp("_Comparer");
        NComparer<OctreeNode,8> cmpr(m_bottom_up, m_top_down);
        cmpr.dump("Comparer result");
        m_timer->stamp("Comparer");
    }
    
    m_raw  = m_ctrl & USE_BOTTOM_UP ? m_bottom_up : m_top_down ; 
    assert(m_raw);

    m_timer->stamp("_SimplifyOctree");
	m_simplified = SimplifyOctree(m_raw, m_threshold);
    m_timer->stamp("SimplifyOctree");
}

void Manager::generateMeshFromOctree(VertexBuffer& vertexBuffer, IndexBuffer& indexBuffer)
{
    assert(m_simplified);

	vertexBuffer.clear();
	indexBuffer.clear();

	GenerateVertexIndices(m_simplified, vertexBuffer, m_bb, m_ce);
	ContourCellProc(m_simplified, indexBuffer);
}



