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

#include "octreedata.h"
#include "octree.h"

#include <cmath>
#include <iostream>
#include <iomanip>


#include "FGLite.h"


typedef std::function<float(float,float,float)> FN ; 


using glm::ivec3 ; 
using glm::vec3 ; 



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


void OctreeNode::DestroyOctree(OctreeNode* node)
{
	if (!node) return ;
	for (int i = 0; i < 8; i++) DestroyOctree(node->children[i]);
	delete node->drawInfo;
	delete node;
}




OctreeNode* OctreeNode::SimplifyOctree(OctreeNode* root, float threshold)
{
    OctreeMgr mgr(root, threshold) ;
    return mgr.simplify();
}


OctreeNode* OctreeMgr::simplify()
{
    OctreeNode* root = simplify_r( m_root, 0 );

    std::cout << "OctreeMgr::simplify"
              << " node_count " << m_node_count
              << " qef_nan " << m_qef_nan
              << " qef_oob " << m_qef_oob
              << std::endl ;

    return root ; 
}


OctreeNode* OctreeMgr::simplify_r(OctreeNode* node, int depth)
{
    m_node_count++ ; 

	if (!node) return NULL;

	if (node->type != OctreeNode::Node_Internal)
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
		node->children[i] = simplify_r(node->children[i], depth+1);
		if (node->children[i])
		{
			OctreeNode* child = node->children[i];
			if (child->type == OctreeNode::Node_Internal)
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

	// at this point the masspoint will actually be a sum, so divide to make it the average
	if (error > m_threshold)
	{
		// this collapse breaches the threshold
		return node;
	}


	// convert to glm vec3 for ease of use
	vec3 position(qefPosition.x, qefPosition.y, qefPosition.z);

    
    //assert(!std::isnan(position.x));
    //assert(!std::isnan(position.y));
    //assert(!std::isnan(position.z));


    bool use_masspoint = false ; 
    if( std::isnan(position.x) || std::isnan(position.y) || std::isnan(position.z) )
    {
        m_qef_nan++ ; 
        use_masspoint = true ; 
    }
	else if (position.x < node->min.x || position.x > (node->min.x + node->size) ||
	   	     position.y < node->min.y || position.y > (node->min.y + node->size) ||
		     position.z < node->min.z || position.z > (node->min.z + node->size) 
            )
	{
        m_qef_oob++ ; 
        use_masspoint = true ; 
	}

    if(use_masspoint)
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
			if (child->type == OctreeNode::Node_Psuedo || 
				child->type == OctreeNode::Node_Leaf)
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
		OctreeNode::DestroyOctree(node->children[i]);
		node->children[i] = NULL ;
	}

	node->type = OctreeNode::Node_Psuedo;
	node->drawInfo = drawInfo;

	return node;
}

// ----------------------------------------------------------------------------



void OctreeNode::GenerateVertexIndices(OctreeNode* node, std::vector<glm::vec3>& vertices, std::vector<glm::vec3>& normals, FGLite* fg)
{
	if (!node)
	{
		return;
	}

	if (node->type != OctreeNode::Node_Leaf)
	{
		for (int i = 0; i < 8; i++)
		{
			GenerateVertexIndices(node->children[i], vertices, normals, fg);
		}
	}

	if (node->type != OctreeNode::Node_Internal)
	{
		OctreeDrawInfo* d = node->drawInfo;
		if (!d)
		{
			printf("Error! Could not add vertex!\n");
			exit(EXIT_FAILURE);
		}

		d->index = vertices.size();

        vec3 pos = d->position ; 

        assert( !std::isnan(pos.x) );
        assert( !std::isnan(pos.y) );
        assert( !std::isnan(pos.z) );

        vec3 world = fg->position_f(pos);

		vertices.push_back(world);
        normals.push_back(d->averageNormal);
	}
}




void OctCheck::report(const char* msg)
{
    std::cout << "OctCheck " << msg 
              << " node_count " << node_count 
              << " bad_node " << bad_node
              << " maxdepth " << maxdepth
              << std::endl 
              ;
} 

void OctCheck::Check( OctreeNode* node, int depth )
{
    node_count++ ; 
    if( depth > maxdepth) maxdepth = depth ; 

	if (node->type != OctreeNode::Node_Internal)
    {
        OctreeDrawInfo* d = node->drawInfo;
        vec3 pos = d->position ; 
   
        bool bad = std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z) ;
        if(bad)
        {
             const glm::ivec3& min = node->min ; 
             std::cout << "OctCheck::Check " 
                       << " node_count " << node_count
                       << " bad_node " << bad_node
                       << " depth " << depth 
                       << " node.min (" << min.x << " " << min.y << " " << min.z << ")" 
                       << " node.size " << node->size 
                       << std::endl ; 

             bad_node++ ;   
        }
    }

    for(int i=0 ; i < 8 ; i++)
    {
        OctreeNode* child = node->children[i] ;  
        if(child) Check(child, depth+1)  ;
    }
}



// ----------------------------------------------------------------------------

void ContourProcessEdge(OctreeNode* node[4], int dir, std::vector<int>& indexBuffer)
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

void ContourEdgeProc(OctreeNode* node[4], int dir, std::vector<int>& indexBuffer)
{
	if (!node[0] || !node[1] || !node[2] || !node[3])
	{
		return;
	}

	if (node[0]->type != OctreeNode::Node_Internal &&
		node[1]->type != OctreeNode::Node_Internal &&
		node[2]->type != OctreeNode::Node_Internal &&
		node[3]->type != OctreeNode::Node_Internal)
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
				if (node[j]->type == OctreeNode::Node_Leaf || node[j]->type == OctreeNode::Node_Psuedo)
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

void ContourFaceProc(OctreeNode* node[2], int dir, std::vector<int>& indexBuffer)
{
	if (!node[0] || !node[1])
	{
		return;
	}

	if (node[0]->type == OctreeNode::Node_Internal || 
		node[1]->type == OctreeNode::Node_Internal)
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
				if (node[j]->type != OctreeNode::Node_Internal)
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
				if (node[order[j]]->type == OctreeNode::Node_Leaf ||
					node[order[j]]->type == OctreeNode::Node_Psuedo)
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

void OctreeNode::ContourCellProc(OctreeNode* node, std::vector<int>& indexBuffer)
{
	if (node == NULL)
	{
		return;
	}

	if (node->type == OctreeNode::Node_Internal)
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



float Density_Func(FGLite* fg, const vec3& offset_ijk)
{
     float fp = fg->value_f(offset_ijk );
     return fp ; 
}

vec3 ApproximateZeroCrossingPosition(const vec3& p0, const vec3& p1, FGLite* fg)
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
		const float density = glm::abs(Density_Func(fg,p));
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

vec3 CalculateSurfaceNormal(const vec3& p, FGLite* fg)
{
	const float H = 0.001f; // hmm delta in ijk-space converted to floats 
	const float dx = Density_Func(fg,p + vec3(H, 0.f, 0.f)) - Density_Func(fg,p - vec3(H, 0.f, 0.f));
	const float dy = Density_Func(fg,p + vec3(0.f, H, 0.f)) - Density_Func(fg,p - vec3(0.f, H, 0.f));
	const float dz = Density_Func(fg,p + vec3(0.f, 0.f, H)) - Density_Func(fg,p - vec3(0.f, 0.f, H));

	return glm::normalize(vec3(dx, dy, dz));
}


int OctreeNode::Corners( const glm::ivec3& leaf_min, FGLite* fg, const int ncorner, const int size )
{
	int corners = 0;
	for (int i = 0; i < ncorner; i++)
	{
		const ivec3 cornerPos = leaf_min + size*CHILD_MIN_OFFSETS[i];
		const float density = Density_Func(fg, vec3(cornerPos));
		const int material = density < 0.f ? MATERIAL_SOLID : MATERIAL_AIR;
		corners |= (material << i);
	}
    return corners ; 
}










void OctreeNode::PopulateLeaf(int corners, OctreeNode* leaf, FGLite* fg)
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

		const vec3 p1 = vec3(leaf->min + CHILD_MIN_OFFSETS[c1]);   // just passing it along 
		const vec3 p2 = vec3(leaf->min + CHILD_MIN_OFFSETS[c2]);  
		const vec3 p = ApproximateZeroCrossingPosition(p1, p2, fg);
		const vec3 n = CalculateSurfaceNormal(p, fg);
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

    const vec3 pos = drawInfo->position ;

    //assert( !std::isnan(pos.x) );

	if (
        pos.x < min.x || 
        pos.x > max.x ||
		pos.y < min.y || 
        pos.y > max.y ||
		pos.z < min.z || 
        pos.z > max.z ||
        std::isnan( pos.x ) || 
        std::isnan( pos.y ) || 
        std::isnan( pos.z )  
       )
	{
		const auto& mp = qef.getMassPoint();
		drawInfo->position = vec3(mp.x, mp.y, mp.z);
	}

	drawInfo->averageNormal = glm::normalize(averageNormal / (float)edgeCount);
	drawInfo->corners = corners;

	leaf->type = OctreeNode::Node_Leaf;
	leaf->drawInfo = drawInfo;

}


OctreeNode* OctreeNode::MakeLeaf(const glm::ivec3& min,  int corners, FGLite* fg, int size )
{
    OctreeNode* leaf = new OctreeNode;
    leaf->size = size ;
    leaf->min = min ; 

    PopulateLeaf( corners, leaf, fg ) ; 

    glm::vec3 pos = leaf->drawInfo->position ;
    assert( !std::isnan(pos.x) ); 
    assert( !std::isnan(pos.y) ); 
    assert( !std::isnan(pos.z) ); 

    return leaf ; 
}


int PopulateLeaves( OctreeNode* node, const int childSize,  FGLite* fg)
{
    int nleaf = 0 ; 
    for (int i = 0; i < 8; i++)
    {
        ivec3 leaf_min = node->min + (CHILD_MIN_OFFSETS[i] * childSize);    //passalong 
        int corners = OctreeNode::Corners(leaf_min, fg );  
        if(corners == 0 || corners == 255) continue ; 

        nleaf++ ; 
        OctreeNode* leaf = OctreeNode::MakeLeaf( leaf_min, corners, fg, childSize );

        node->children[i] = leaf ;
    } 
    return nleaf ; 
}


OctreeNode* OctreeNode::ConstructOctreeNodes(OctreeNode* node, FGLite* fg, int& count)
{
    assert(node && node->size > 1);
	bool hasChildren = false;

	const int childSize = node->size / 2;
    if(childSize == 1)
    {
        int nleaf = PopulateLeaves(node, childSize, fg);
        hasChildren = nleaf > 0 ; 
        count += 8 ; // candidate leaves tested   
    } 
    else
    {
        int nchild = 0 ;  
        for (int i = 0; i < 8; i++)
        {
            ivec3 child_min = node->min + (CHILD_MIN_OFFSETS[i] * childSize) ;  //passalong

            OctreeNode* child = new OctreeNode;
            child->size = childSize;
            child->min = child_min ; 
            child->type = OctreeNode::Node_Internal;

            OctreeNode* confirmed = ConstructOctreeNodes(child, fg, count);
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




