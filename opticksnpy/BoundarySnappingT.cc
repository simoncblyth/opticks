// /usr/local/opticks/externals/openflipper/OpenFlipper-3.1/Plugin-MeshRepair/BoundarySnappingT.cc
/*===========================================================================*\
 *                                                                           *
 *                               OpenMesh                                    *
 *           Copyright (c) 2001-2015, RWTH-Aachen University                 *
 *           Department of Computer Graphics and Multimedia                  *
 *                          All rights reserved.                             *
 *                            www.openflipper.org                            *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * This file is part of OpenFlipper.                                         *
 *---------------------------------------------------------------------------*
 *                                                                           *
 * Redistribution and use in source and binary forms, with or without        *
 * modification, are permitted provided that the following conditions        *
 * are met:                                                                  *
 *                                                                           *
 * 1. Redistributions of source code must retain the above copyright notice, *
 *    this list of conditions and the following disclaimer.                  *
 *                                                                           *
 * 2. Redistributions in binary form must reproduce the above copyright      *
 *    notice, this list of conditions and the following disclaimer in the    *
 *    documentation and/or other materials provided with the distribution.   *
 *                                                                           *
 * 3. Neither the name of the copyright holder nor the names of its          *
 *    contributors may be used to endorse or promote products derived from   *
 *    this software without specific prior written permission.               *
 *                                                                           *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *
 *                                                                           *
 \*===========================================================================*/

/*===========================================================================*\
 *                                                                           *
 *   $Revision$                                                         *
 *   $Date$                   *
 *                                                                           *
 \*===========================================================================*/

/** \file BoundarySnappingT.cc
 */

//=============================================================================
//
//  CLASS MeshFixing - IMPLEMENTATION
//
//=============================================================================

#define BOUNDARYSNAPPING_CC

//== INCLUDES =================================================================

#include <map>
#include "BoundarySnappingT.hh"

//== NAMESPACE ===============================================================


//== IMPLEMENTATION ==========================================================


template<class MeshT>
BoundarySnappingT<MeshT>::BoundarySnappingT(MeshT& _mesh ) :
mesh_(_mesh)
{
}



template<typename MeshT>
//bool sort_less_pair_second(const VHD &lhs,const VHD &rhs)
bool sort_less_pair_second(const std::pair<typename MeshT::VertexHandle,double> &lhs,const std::pair<typename MeshT::VertexHandle,double> &rhs)
{
  return lhs.second < rhs.second;
}

template<class MeshT>
void BoundarySnappingT<MeshT>::snap(double _epsilon)
{
  std::vector<VH> v_boundary;

  // collect all boundary vertices
  for (typename MeshT::VertexIter v_iter = mesh_.vertices_begin(); v_iter != mesh_.vertices_end(); ++v_iter)
    if (mesh_.is_boundary(*v_iter) && mesh_.status(*v_iter).selected())
      v_boundary.push_back(*v_iter);

  //two maps
  // vertexDistMap saves the vertex and his distances as pairs
  // vertexVertexMap saves the vertices of a vertex in range and the distances
  std::vector< VHD > vertexDistMap;
  std::map<VH,std::vector<VHD > > vertexVertexMap;

  // get all boundary vertices in range and save them into the maps
  for (VHI oIter = v_boundary.begin(); oIter != v_boundary.end(); ++oIter)
  {
    typename MeshT::Point pos = mesh_.point(*oIter);
    if (!mesh_.status(*oIter).deleted())
    {
      std::vector< VHD > verticesInRange;

      //collect all vertices in range
      for (VHI cIter = v_boundary.begin(); cIter != v_boundary.end(); ++cIter)
      {
        if ( !mesh_.status(*cIter).deleted() && cIter != oIter)
        {
          double dist = (pos - mesh_.point(*cIter)).length();

          if ( dist <= _epsilon )
            verticesInRange.push_back(std::make_pair(*cIter,dist));
        }
      }

      // sort them, so nearest vertex is on position 0 (if exist)
      if (!verticesInRange.empty())
      {
        std::sort( verticesInRange.begin(),verticesInRange.end(),&sort_less_pair_second<MeshT> );
        vertexDistMap.push_back(std::make_pair(*oIter,verticesInRange[0].second));
        vertexVertexMap[*oIter] = verticesInRange;
      }
    }
  }

  // vertexDistMap
  //      vector of (vh,dist) pairs, where dist is distance to another vh 
  // vertexVertexMap
  //      map of vh->vertexDistMap
  //
  // above calulated distances of all vertex pairs within epsilon of each other and 
  // sorted them ... providing a map from each vertex to a vector of other verts and 
  // distances  

  bool finished = false;
  while(!finished)
  {
    finished = true;

    double min = _epsilon;
    VH v_old;//will be replaced by v_new
    VH v_new;
    VHDI v_oldIter = vertexDistMap.end();
    VHDI v_newIter;

    // find next min pair
    for (VHDI vd_iter = vertexDistMap.begin(); vd_iter != vertexDistMap.end(); ++vd_iter)
    {
      VH v_1 = vd_iter->first;
      if (v_1.is_valid() && !mesh_.status(v_1).deleted() && vertexVertexMap.find(v_1) != vertexVertexMap.end())
      {
        VH v_2;
        std::vector<VHD >& verticesInRange = vertexVertexMap[v_1];

        for (VHDI iter = verticesInRange.begin(); iter != verticesInRange.end(); ++iter)
        {
          //check if v_2 shares a face with v_1
          //if so, it is not usable
          v_2 = iter->first;

          for(MVFI vf_iter = mesh_.vf_begin(v_1); vf_iter.is_valid() && v_2.is_valid(); ++vf_iter)
            for (MFVI fv_iter = mesh_.fv_begin(*vf_iter); fv_iter.is_valid() && v_2.is_valid(); ++fv_iter)
              if (*fv_iter == v_2)
                v_2 = VH();

          const bool validPair = v_2.is_valid() && !mesh_.status(v_2).deleted() && mesh_.is_boundary(v_2);

          //if v_2 is valid, save it, or erase it if not, because v_2 will not be valid in the future
          if (validPair && iter->second <= min)
          {
            //new min pair found
            min = iter->second;
            v_old = v_1;
            v_new = v_2;
            finished = false;
            v_oldIter = vd_iter;
            v_newIter = iter;
          }
        }
      }

    }
    // merge, if not finished (pair found)
    if (!finished)
    {
      //remove the vertex since we will proceed with it
      vertexVertexMap[v_old].erase(v_newIter);
      //save all faces, because faces will be added/deleted
      std::vector<FH> faces;
      for (MVFI vf_iter = mesh_.vf_begin(v_old); vf_iter.is_valid(); ++vf_iter)
        if (!mesh_.status(*vf_iter).deleted())
          faces.push_back(*vf_iter);

      //replace v_old with v_new by creating new faces with v_new instead of v_old if possible
      for (FHI f_iter = faces.begin(); f_iter !=faces.end(); ++f_iter)
      {
        FH fHandle = *f_iter;
        if (!fHandle.is_valid() || mesh_.status(fHandle).deleted())
          continue;

        //get face vertices
        std::vector<VH> f_vertices;
        for(MFVI fv_iter = mesh_.fv_begin(fHandle); fv_iter.is_valid(); ++fv_iter)
          f_vertices.push_back( *fv_iter );

        mesh_.delete_face(fHandle,false);

        //try to add new face : with v_old replaced by v_new
        std::vector<VH> newFace_vertices(f_vertices);
        std::replace(newFace_vertices.begin(),newFace_vertices.end(),v_old,v_new);  
        
        FH faceH = mesh_.add_face(newFace_vertices);

        if (!faceH.is_valid())
        {
          //failed, try reverse direction
          std::reverse(newFace_vertices.begin(),newFace_vertices.end());
          faceH = mesh_.add_face(newFace_vertices);
          if (!faceH.is_valid())
          {
            //failed, so add the old one
            mesh_.add_face(f_vertices);
          }
        }
      }
    }
    vertexDistMap.erase(v_oldIter);

    //todo: delete vertex before proceed. Now, they will be deleted at the end resulting worse snap
  }
  mesh_.delete_isolated_vertices();

  mesh_.garbage_collection();

}
