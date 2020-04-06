/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include <iostream>
// -------------------- OpenMesh
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
// ----------------------------------------------------------------------------
typedef OpenMesh::PolyMesh_ArrayKernelT<>  MyMesh;
// ----------------------------------------------------------------------------
// Build a simple cube and write it to std::cout
  
int main(int argc, char** argv)
{
  MyMesh mesh;
  // generate vertices
  MyMesh::VertexHandle vhandle[8];
  vhandle[0] = mesh.add_vertex(MyMesh::Point(-1, -1,  1));
  vhandle[1] = mesh.add_vertex(MyMesh::Point( 1, -1,  1));
  vhandle[2] = mesh.add_vertex(MyMesh::Point( 1,  1,  1));
  vhandle[3] = mesh.add_vertex(MyMesh::Point(-1,  1,  1));
  vhandle[4] = mesh.add_vertex(MyMesh::Point(-1, -1, -1));
  vhandle[5] = mesh.add_vertex(MyMesh::Point( 1, -1, -1));
  vhandle[6] = mesh.add_vertex(MyMesh::Point( 1,  1, -1));
  vhandle[7] = mesh.add_vertex(MyMesh::Point(-1,  1, -1));
  // generate (quadrilateral) faces
  std::vector<MyMesh::VertexHandle>  face_vhandles;
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[0]);
  face_vhandles.push_back(vhandle[1]);
  face_vhandles.push_back(vhandle[2]);
  face_vhandles.push_back(vhandle[3]);
  mesh.add_face(face_vhandles);
 
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[7]);
  face_vhandles.push_back(vhandle[6]);
  face_vhandles.push_back(vhandle[5]);
  face_vhandles.push_back(vhandle[4]);
  mesh.add_face(face_vhandles);
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[1]);
  face_vhandles.push_back(vhandle[0]);
  face_vhandles.push_back(vhandle[4]);
  face_vhandles.push_back(vhandle[5]);
  mesh.add_face(face_vhandles);
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[2]);
  face_vhandles.push_back(vhandle[1]);
  face_vhandles.push_back(vhandle[5]);
  face_vhandles.push_back(vhandle[6]);
  mesh.add_face(face_vhandles);
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[3]);
  face_vhandles.push_back(vhandle[2]);
  face_vhandles.push_back(vhandle[6]);
  face_vhandles.push_back(vhandle[7]);
  mesh.add_face(face_vhandles);
  face_vhandles.clear();
  face_vhandles.push_back(vhandle[0]);
  face_vhandles.push_back(vhandle[3]);
  face_vhandles.push_back(vhandle[7]);
  face_vhandles.push_back(vhandle[4]);
  mesh.add_face(face_vhandles);
  // write mesh to output.obj


  const char* path = "/tmp/UseOpenMesh.off" ; 
  std::cout << argv[0] << " writing to " << path << std::endl ; 

  try
  {
    if ( !OpenMesh::IO::write_mesh(mesh, path) )
    {
      std::cerr << "Cannot write mesh to file 'output.off'" << std::endl;
      return 1;
    }
  }
  catch( std::exception& x )
  {
    std::cerr << x.what() << std::endl;
    return 1;
  }
  return 0;
}
