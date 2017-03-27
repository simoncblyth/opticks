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


//#include	"density.h"

#include "Timer.hpp"
#include "GLMFormat.hpp"
#include "PLOG.hh"
//#include "mortonlib/morton3d.h"
#include  <boost/unordered_map.hpp>

#include "NBBox.hpp"
#include "NGrid3.hpp"
#include "NField3.hpp"



float Density_Func(std::function<float(float,float,float)>* f, const vec4& ce, const vec3& ijk)
{
    vec3 p = vec3(ce) + ijk*ce.w ; 

    float fp = (*f)(p.x, p.y, p.z);

  /*
   printf(" ce:(%10.3f %10.3f %10.3f %10.3f) ijk: (%10.3f %10.3f %10.3f) p:(%10.3f %10.3f %10.3f)  -> %10.3f \n", 
         ce.x,ce.y,ce.z, ce.w,
         ijk.x,ijk.y,ijk.z, 
         p.x,p.y,p.z,  fp );
   */

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

/*
const int edgemask[3] = { 5, 3, 6 } ;

const int vertMap[8][3] = 
{
	{0,0,0},
	{0,0,1},
	{0,1,0},
	{0,1,1},
	{1,0,0},
	{1,0,1},
	{1,1,0},
	{1,1,1}
};

const int faceMap[6][4] = {{4, 8, 5, 9}, {6, 10, 7, 11},{0, 8, 1, 10},{2, 9, 3, 11},{0, 4, 2, 6},{1, 5, 3, 7}} ;
*/


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

void GenerateVertexIndices(OctreeNode* node, VertexBuffer& vertexBuffer, const nbbox& bb)
{
	if (!node)
	{
		return;
	}

	if (node->type != Node_Leaf)
	{
		for (int i = 0; i < 8; i++)
		{
			GenerateVertexIndices(node->children[i], vertexBuffer, bb);
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
        //vec3 xyz = vec3(ce) + ijk*ce.w ; 
        vec3 xyz = bb.min + ijk*bb.side  ; 

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

// ----------------------------------------------------------------------------

vec3 ApproximateZeroCrossingPosition(const vec3& p0, const vec3& p1, std::function<float(float,float,float)>* f, const vec4& ce)
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

vec3 CalculateSurfaceNormal(const vec3& p, std::function<float(float,float,float)>* f, const vec4& ce)
{
	const float H = 0.001f; // hmm delta in ijk-space converted to floats 
	const float dx = Density_Func(f,ce,p + vec3(H, 0.f, 0.f)) - Density_Func(f,ce,p - vec3(H, 0.f, 0.f));
	const float dy = Density_Func(f,ce,p + vec3(0.f, H, 0.f)) - Density_Func(f,ce,p - vec3(0.f, H, 0.f));
	const float dz = Density_Func(f,ce,p + vec3(0.f, 0.f, H)) - Density_Func(f,ce,p - vec3(0.f, 0.f, H));

	return glm::normalize(vec3(dx, dy, dz));
}

// ----------------------------------------------------------------------------


int Corners( const ivec3& leaf_min, std::function<float(float,float,float)>* f, const vec4& ce, const int ncorner=8, const int size=1 )
{
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






void PopulateLeaf(int corners, OctreeNode* leaf, std::function<float(float,float,float)>* f, const vec4& ce )
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


OctreeNode* ConstructLeaf(OctreeNode* leaf, std::function<float(float,float,float)>* f, const vec4& ce )
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



// -------------------------------------------------------------------------------

OctreeNode* ConstructOctreeNodes_0(OctreeNode* node, std::function<float(float,float,float)>* f, const vec4& ce)
{
	if (!node)
	{
		return nullptr;
	}

	if (node->size == 1)
	{
		return ConstructLeaf(node, f, ce);
	}
	
	const int childSize = node->size / 2;
	bool hasChildren = false;

	for (int i = 0; i < 8; i++)
	{
		OctreeNode* child = new OctreeNode;
		child->size = childSize;
		child->min = node->min + (CHILD_MIN_OFFSETS[i] * childSize);
		child->type = Node_Internal;

		node->children[i] = ConstructOctreeNodes_0(child, f, ce);
		hasChildren |= (node->children[i] != nullptr);
	}

	if (!hasChildren)
	{
		delete node;
		return nullptr;
	}

	return node;
}



bool HasLeaves( const ivec3& min, const int leafSize, std::function<float(float,float,float)>* f, const vec4& ce )
{
    assert(leafSize == 1);
    for (int i = 0; i < 8; i++)
    {
        ivec3 leaf_min = min + (CHILD_MIN_OFFSETS[i] * leafSize); 
        int corners = Corners(leaf_min, f, ce );  
        if(corners == 0 || corners == 255) continue ;  // not leaf, keep looking
        return true ;    // found one, so early exit 
    }
    return false ; 
}


int PopulateLeaves( OctreeNode* node, const int childSize,  std::function<float(float,float,float)>* f, const vec4& ce)
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



bool HasChildren( const ivec3& min , const int nodeSize, std::function<float(float,float,float)>* f, const vec4& ce)
{
	const int childSize = nodeSize / 2;
    if(childSize == 1) return HasLeaves( min, childSize, f, ce );
    for (int i = 0; i < 8; i++)
    {
         ivec3 child_min = min + (CHILD_MIN_OFFSETS[i] * childSize) ;
         bool has = HasChildren(child_min, childSize, f, ce ); 
         if(!has) continue ; 
         return true ; 
    }
    return false ; 
}


// -------------------------------------------------------------------------------

OctreeNode* ConstructOctreeNodes(OctreeNode* node, std::function<float(float,float,float)>* f, const vec4& ce, int& count)
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



// -------------------------------------------------------------------------------

void DumpIjk( const int idx, const ivec3& ijk, std::function<float(float,float,float)>* f, const vec4& ce )
{
    /*
    // hmm ce.w is not real extent, by jiggery pokered ratio of gridsize and world size
    nvec3 fmin = make_nvec3( ce.x - ce.w , ce.y - ce.w, ce.z - ce.w );
    nvec3 fmax = make_nvec3( ce.x + ce.w , ce.y + ce.w, ce.z + ce.w );
    NField3 field( f , fmin, fmax );
    */

    const vec3 xyz = vec3(ce) + vec3(ijk)*ce.w ; 
    float fxyz = Density_Func( f, ce, ijk );

    std::cout << " idx " << std::setw(5) << idx 
              << " ijk (" 
                   << std::setw(3) << ijk.x 
                   << "," << std::setw(3) << ijk.y 
                   << "," << std::setw(3) << ijk.z 
              << ")"     
              << " xyz (" 
                   << std::setw(10) << xyz.x 
                   << "," << std::setw(10) << xyz.y 
                   << "," << std::setw(10) << xyz.z 
              << ")"     
              << " --> "
              << std::setw(10) << fxyz 
              << std::endl ; 
}  


void CheckDomain( const ivec3& min, const int level, std::function<float(float,float,float)>* f, const vec4& ce )
{
    int size = 1 << level ; 
    std::cout << "CheckDomain "
              << " size " << size
              << " min (" << min.x << "," << min.y << "," << min.z << ")"
              << " ce  (" << ce.x << "," << ce.y << "," << ce.z << "," << ce.w << ")"             
              << std::endl ; 
        
	for (int i = 0; i < 8; i++)
    {
        const ivec3 ijk = min + CHILD_MIN_OFFSETS[i] * size ; 
        DumpIjk( i, ijk, f, ce );
	}
}


class Constructor {
    typedef boost::unordered_map<unsigned, OctreeNode*> UMAP ;
    UMAP cache[10] ; 
    public:
        Constructor(int level, NField3* field);
        OctreeNode* create();
    private:
        int         m_level ; 
        NGrid3*     m_grid ; 
        NField3*    m_field ; 
        OctreeNode* m_root ; 
};


Constructor::Constructor(int level, NField3* field)
   :
   m_level(level),  
   m_grid(new NGrid3(level)),
   m_field(field),
   m_root(NULL)
{
}

OctreeNode* Constructor::create()
{
    for(int c=0 ; c < m_grid->nloc ; c++) 
    {



    }
    return NULL ; 
}



OctreeNode* ConstructOctreeBottomUp( const ivec3& min, const int n_level, std::function<float(float,float,float)>* f, const vec4& ce )
{
    static const int maxlevel = 10 ; 
    assert( n_level >= 0 && n_level < maxlevel ); 

    const unsigned coarse_ = 1 ; // coarse resolution, 1 or 2-levels up from nominal level 

    NMultiGrid3 mg ; 
    const NGrid3& nominal = *mg.grid[n_level] ;
    const NGrid3& coarse  = *mg.grid[n_level-coarse_] ;
    const NGrid3& subtile = *mg.grid[coarse_];

    nvec3 fmin = make_nvec3( ce.x - ce.w , ce.y - ce.w, ce.z - ce.w );
    nvec3 fmax = make_nvec3( ce.x + ce.w , ce.y + ce.w, ce.z + ce.w );
    NField3 field( f , fmin, fmax );


    std::cout << "ConstructOctreeBottomUp"
              << " nominal " << nominal.desc()
              << " coarse " << coarse.desc()
              << " subtile " << subtile.desc()
              << " min " << gformat(min)
              << std::endl ; 

    ivec3 leaf_min ; 
    ivec3 coarse_min ; 
    ivec3 d_min ; 

    typedef boost::unordered_map<unsigned, OctreeNode*> UMAP ;
    UMAP cache[10] ; 

    unsigned num_leaf(0); 
    unsigned num_from_cache(0); 
    unsigned num_into_cache(0); 

    for(int c=0 ; c < coarse.nloc ; c++) 
    {
        nivec3 c2n_ijk= coarse.ijk( c );
        c2n_ijk *= subtile.size ;    // scale coarse coordinates up to nominal 

        coarse_min = min ;
        coarse_min.x += c2n_ijk.x ; 
        coarse_min.y += c2n_ijk.y ; 
        coarse_min.z += c2n_ijk.z ; 

        // nominal.size 128   
        // coarse.size   64

        nvec3 cpos = coarse.fpos(c); 
        int corners = Corners( coarse_min, f, ce, 8, subtile.size ); 
        int corners2 = field.zcorners(cpos, nominal.elem*subtile.size ) ; 

        if(corners != corners2)
            std::cout 
                 << " corners 0b" << std::bitset<8>(corners) 
                 << " corners2 0b" << std::bitset<8>(corners2)
                 << " cpos " << cpos.desc()
                 << " nominal.elem " << nominal.elem
                 << " subtile.size " << subtile.size
                 << std::endl ;  


        assert(corners == corners2);


        if(corners == 0 || corners == 255) continue ;    // all in/out not interesting
 
        for(int s=0 ; s < subtile.nloc ; s++)  // over nominal(at level) voxels in coarse tile
        {
            nivec3 s_ijk = subtile.ijk( s );
 
            leaf_min = coarse_min ;
            leaf_min.x += s_ijk.x ; 
            leaf_min.y += s_ijk.y ; 
            leaf_min.z += s_ijk.z ; 

            int corners = Corners( leaf_min, f, ce, 8, nominal.size );  // <-- this dominates time to construct nodes
            if(corners == 0 || corners == 255) continue ;    // all in/out not interesting

            num_leaf++ ; 

            OctreeNode* leaf = new OctreeNode;
            leaf->size = 1 ;
            leaf->min = leaf_min ; 

            PopulateLeaf( corners, leaf, f, ce ) ; 


           // nominal absolute morton, not the tile relative c, and need to get rid of the min (-64,-64,-64...)
            unsigned dloc = nominal.loc( leaf_min.x - min.x , leaf_min.y - min.y, leaf_min.z - min.z );  

            int depth = n_level ; // start from nominal level, with the leaves 
            unsigned dsize = 1 ; 
            unsigned dchild = dloc & 7 ;  // lowest 3 bits, gives child index in immediate parent

            OctreeNode* node = leaf ; 
            OctreeNode* dnode = NULL ; 

            while(depth >= 1)
            {
                // decrement depth, right-shift morton code to that of parent, left shift size doubling it 
                depth-- ; 
                dloc >>= 3 ;     
                dsize <<= 1 ;     

                const NGrid3& dgrid = *mg.grid[depth] ; 

                UMAP::const_iterator it = cache[depth].find(dloc);

                if(it == cache[depth].end())
                {
                    num_into_cache++ ; 

                    nivec3 d_ijk = dgrid.ijk(dloc); 
                    d_ijk *= dsize ; // scale to nominal 

                    d_min = min ; 
                    d_min.x += d_ijk.x ; 
                    d_min.y += d_ijk.y ; 
                    d_min.z += d_ijk.z ; 

                    if(num_into_cache < 10)
                    std::cout << "into_cache " 
                              << " num_into_cache " << num_into_cache
                              << " dloc " << dloc
                              << " d_ijk " << d_ijk.desc()
                              << " dsize " << dsize
                              << " d_min " << gformat(d_min)
                              << std::endl ; 
                    

                    dnode = new OctreeNode ; 
                    dnode->size = dsize ; 
                    dnode->min = d_min ; 
	                dnode->type = Node_Internal;

                    cache[depth].emplace(dloc, dnode)  ;
                }
                else
                {
                    num_from_cache++ ; 
                    dnode = it->second ;     
                }
 
                dnode->children[dchild] = node ;  
                node = dnode ; 
                dchild = dloc & 7 ;  // child index for next round
 
            }              // up the heirarchy from each leaf to root
        }                  // over nominal voxels within coarse tile
    }                      // over coarse tiles
    

    UMAP::const_iterator it0 = cache[0].find(0);
    OctreeNode* root = it0 == cache[0].end() ? NULL : it0->second ; 

    std::cout << "ConstructOctreeBottomUp"
              << " num_leaf " << num_leaf 
              << " num_into_cache " << num_into_cache 
              << " num_from_cache " << num_from_cache 
              << " num_leaf/nominal.nloc " << float(num_leaf)/float(nominal.nloc) 
              << std::endl ;

    assert(root);
    return root ; 
}





class Traverser {
   public:
       Traverser(OctreeNode* root, const char* msg="Traverser", int verbosity=0, int nodelimit=30) 
           : 
           m_root(root), 
           m_nodecount(0),
           m_nodelimit(nodelimit),
           m_verbosity(verbosity)  
       {
           traverse_r(m_root, 0);
           std::cout 
                << msg 
                << " nodecount " << m_nodecount 
                << std::endl ; 
       }
   private:
       void traverse_r(OctreeNode* node, int depth)
       {
           m_nodecount++ ; 

           if(m_verbosity > 0 && m_nodecount < m_nodelimit)
           {
               std::cout << "traverse_r" 
                         << " depth " << depth 
                         << " nodecount " << m_nodecount
                         << " nodelimit " << m_nodelimit 
                         << " size " << node->size
                         << " min " << gformat(node->min) 
                         << std::endl ; 
           } 

           for(int i=0 ; i < 8 ; i++)
           {
               OctreeNode* child = node->children[i] ;
               if(child) traverse_r(child, depth+1 );
           }
       }
   private:
       OctreeNode* m_root ; 
       int         m_nodecount ; 
       int         m_nodelimit ; 
       int         m_verbosity ; 
};



class Comparer {
       enum {
          NSTATE=4 
       };
   public:
      Comparer(OctreeNode* a, OctreeNode* b)
          :
          m_a(a),
          m_b(b),
          m_same(false),
          m_equals_count(0),
          m_content_count(0)
       {
           for(int i=0 ; i < NSTATE ; i++) m_state[i] = 0 ; 
           m_same = equals(m_a, m_b);
       }

      void dump(const char* msg)
      {
          LOG(info) << msg << " " << ( m_same ? "ARE-SAME" : "NOT-SAME" ) ; 
          int state_tot = 0 ; 
          for(int i=0 ; i < NSTATE ; i++)  
          {
             std::cout << " state " <<  i << " : " << m_state[i] << std::endl ; 
             state_tot += m_state[i] ; 
          }
          std::cout 
                  << " equals_count " << m_equals_count 
                  << " content_count " << m_content_count 
                  << " state_tot " << state_tot 
                  << std::endl ;   
      }  

       bool equal_content( OctreeNode* a, OctreeNode* b)
       { 
            m_content_count++ ;  
            return 
                   a->type == b->type && 
                   a->size == b->size &&
                   a->min == b->min 
                   ;
       }

       bool equals( OctreeNode* a, OctreeNode* b)
       {
           int ab = ((!!a) << 1 ) | (!!b) ;   //  !!(NULL) -> 0, !!(non-NULL) -> 1
           m_state[ab]++ ;   
           m_equals_count++ ; 

           if( ab < 3)
           {
               return ab == 0 ;  // 0:both NULL, 1/2 one NULL other non-NULL
           } 
           else
           {
               int num_equal = 0 ; 
               for(int i=0 ; i < 8 ; i++) num_equal += equals( a->children[i], b->children[i] ) ;
               return num_equal == 8 && equal_content(a, b) ; 
           }
       }

   private:
       OctreeNode* m_a ; 
       OctreeNode* m_b ;
       bool        m_same ; 
       int         m_state[NSTATE] ; 
       int         m_equals_count ; 
       int         m_content_count ; 

}; 





 






OctreeNode* BuildOctree(const ivec3& min, const int level, const float threshold, std::function<float(float,float,float)>* f, const nbbox& bb, Timer* timer)
{
    nvec4     bbce = bb.center_extent();

    float xyzExtent = bbce.w  ;
    float ijkExtent = fabs(min.x) ;
    float ijk2xyz = xyzExtent/ijkExtent ;   // octree -> real world coordinates

    glm::vec4 ce(bbce.x, bbce.y, bbce.z, ijk2xyz );

    LOG(info) << "BuildOctree"
              << " xyzExtent " << xyzExtent
              << " ijkExtent " << ijkExtent
              << " bbce " << bbce.desc()
              << " ce " << gformat(ce)
              << " ilow " << gformat(m_ilow)
              ;


    int size = 1 << level ; 
    CheckDomain( min, level, f, ce ); 

    enum { 
           BUILD_BOTTOM_UP = 0x1 << 0, 
           BUILD_TOP_DOWN  = 0x1 << 1,
           USE_BOTTOM_UP   = 0x1 << 2, 
           USE_TOP_DOWN    = 0x1 << 3, 
           BUILD_BOTH      = BUILD_BOTTOM_UP | BUILD_TOP_DOWN
         };


    //unsigned ctrl = BUILD_BOTH | USE_BOTTOM_UP ; 
    //unsigned ctrl = BUILD_BOTH | USE_TOP_DOWN ; 
    unsigned ctrl = BUILD_BOTTOM_UP | USE_BOTTOM_UP ; 

    OctreeNode* bottom_up = NULL ; 
    OctreeNode* top_down = NULL ; 

    if( ctrl & BUILD_BOTTOM_UP )
    {
        timer->stamp("_ConstructOctreeBottomUp");
        bottom_up = ConstructOctreeBottomUp( min, level, f, ce );
        assert(bottom_up);
        timer->stamp("ConstructOctreeBottomUp");
        Traverser(bottom_up, "bottom_up", 1, 30 );
    }

    if( ctrl & BUILD_TOP_DOWN )
    {
  	    OctreeNode* root0 = new OctreeNode;
  	    root0->min = min;
	    root0->size = size;
	    root0->type = Node_Internal;

        timer->stamp("_ConstructOctreeNodes");
        int count = 0 ; 
	    top_down = ConstructOctreeNodes(root0, f, ce, count);
        timer->stamp("ConstructOctreeNodes");
        std::cout << "ConstructOctreeNodes count " << count << std::endl ; 
        Traverser(top_down, "top_down", 1, 30);
    }

    if( ctrl & BUILD_BOTH )
    {
        timer->stamp("_Comparer");
        Comparer cmpr(bottom_up, top_down);
        cmpr.dump("Comparer result");
        timer->stamp("Comparer");
    }

    
    OctreeNode* root = ctrl & USE_BOTTOM_UP ? bottom_up : top_down ; 
    assert(root);

    timer->stamp("_SimplifyOctree");
	OctreeNode* result = SimplifyOctree(root, threshold);
    timer->stamp("SimplifyOctree");

	return result  ;
}

// ----------------------------------------------------------------------------

void GenerateMeshFromOctree(OctreeNode* node, VertexBuffer& vertexBuffer, IndexBuffer& indexBuffer, const nbbox& bb)
{
	if (!node)
	{
		return;
	}

	vertexBuffer.clear();
	indexBuffer.clear();

	GenerateVertexIndices(node, vertexBuffer, bb);
	ContourCellProc(node, indexBuffer);
}

// -------------------------------------------------------------------------------

void DestroyOctree(OctreeNode* node)
{
	if (!node)
	{
		return;
	}

	for (int i = 0; i < 8; i++)
	{
		DestroyOctree(node->children[i]);
	}

	if (node->drawInfo)
	{
		delete node->drawInfo;
	}

	delete node;
}

// -------------------------------------------------------------------------------
