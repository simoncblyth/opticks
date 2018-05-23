// /usr/local/opticks/externals/openflipper/OpenFlipper-3.1/Plugin-MeshRepair/BoundarySnappingT.hh
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


#ifndef BOUNDARYSNAPPING_HH
#define BOUNDARYSNAPPING_HH

//=============================================================================
//
//  CLASS BoundarySnappingT
//
//=============================================================================

//== INCLUDES =================================================================

//== NAMESPACE ================================================================

//== CLASS DEFINITION =========================================================

/** \brief Snaps selected vertices at boundaries
 *
 * Snaps selected boundary vertices together if they are closer than the given
 * distance. No new vertices will be introduced on either edge, so they are just
 * snapped to existing ones.
 *
 * If vertices in the interior are selected, they will also get snapped to the
 * opposite boundary, if in range.
 */

#include <vector>

template<class MeshT>
class BoundarySnappingT {

public:
  // SCB: use typedefs to un-obfuscate the code
  typedef typename MeshT::VertexHandle VH ; 
  typedef std::pair<VH,double>         VHD ; 

  typedef typename std::vector< VH >::iterator VHI ;  
  typedef typename std::vector<VHD >::iterator  VHDI ; 
  typedef typename MeshT::FaceHandle   FH ; 

  typedef typename std::vector<FH>::iterator FHI ; 

  typedef typename MeshT::VertexFaceIter  MVFI ;  
  typedef typename MeshT::FaceVertexIter  MFVI ; 




  BoundarySnappingT(MeshT& _mesh );

  /** \brief snaps boundary vertices
   *
   * snaps selected boundary vertices where the vertices
   * distance is not greater than the given distance
   *
   * @param _epsilon max Distance between 2 boundary vertices
   *
   */
  void snap(double _epsilon);

private:

  MeshT& mesh_;

};

#if defined(INCLUDE_TEMPLATES) && !defined(BOUNDARYSNAPPING_CC)
#define BOUNDARYSNAPPING_TEMPLATES
#include "BoundarySnappingT.cc"
#endif

#endif

