//  https://www.openmesh.org/media/Documentations/OpenMesh-4.1-Documentation/a00058.html
//  https://mailman.rwth-aachen.de/pipermail/openmesh/2009-August/000305.html

#include <iostream>
#include <iomanip>

//
// without the bookends this asserts on attempting to delete a face
// when the -fvisibility=hidden compiler option is in use
//
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"

#pragma GCC visibility push(default)
#endif


#ifdef _MSC_VER
#pragma warning( push )
// OpenMesh/Core/Mesh/AttribKernelT.hh(140): warning C4127: conditional expression is constant
#pragma warning( disable : 4127 )
// OpenMesh/Core/Utils/vector_cast.hh(94): warning C4100: '_dst': unreferenced formal parameter
#pragma warning( disable : 4100 )
// openmesh\core\utils\property.hh(156): warning C4702: unreachable code  
#pragma warning( disable : 4702 )
#endif


#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/System/config.h>
#include <OpenMesh/Core/Mesh/Status.hh>


#ifdef _MSC_VER
#pragma warning( pop )
#endif



#ifdef __clang__
#pragma GCC visibility pop

#pragma clang diagnostic pop
#endif




// follow the old documentation, not using the dy
// https://www.openmesh.org/media/Documentations/OpenMesh-2.1-Documentation/tutorial_07b.html

#ifdef NEW_WAY
struct MyTraits : public OpenMesh::DefaultTraits
{
};
#else
struct MyTraits : public OpenMesh::DefaultTraits
{
  VertexAttributes(OpenMesh::Attributes::Status);
  FaceAttributes(OpenMesh::Attributes::Status);
  EdgeAttributes(OpenMesh::Attributes::Status);
};
#endif


typedef OpenMesh::PolyMesh_ArrayKernelT<MyTraits>  MyMesh;
// Build a simple cube and delete it except one face


template <typename MeshT>
void dump(MeshT* mesh, const char* msg="dump")
{
    typedef typename MeshT::Point P ; 
    typedef typename MeshT::VertexHandle VH ; 
    typedef typename MeshT::FaceHandle FH ; 
    typedef typename MeshT::ConstFaceVertexIter FVI ; 

    unsigned nv = mesh->n_vertices();
    unsigned nf = mesh->n_faces();
   

    std::cerr << msg 
              << " nv " << nv  
              << " nf " << nf  
              << std::endl ; 
 
    for(unsigned i=0 ; i < nv ; i++) 
    {
        VH v = mesh->vertex_handle(i) ;
        P p = mesh->point(v);
//      P n = mesh->normal(v);

        std::cerr << " i " << std::setw(4) << i 
                  << " p " 
                  << std::setw(10) << p[0]
                  << std::setw(10) << p[1]
                  << std::setw(10) << p[2]
/*
                  << " n " 
                  << std::setw(10) << n[0]
                  << std::setw(10) << n[1]
                  << std::setw(10) << n[2]
*/
                  << std::endl ; 

    }

    for(unsigned int i=0 ; i < nf ; i++)
    {   
        FH fh = mesh->face_handle(i) ;   

        std::stringstream ssf ; 

        unsigned j(0) ;   
        for(FVI fv=mesh->cfv_iter(fh) ; fv.is_valid() ; fv++ )
        {   
            ssf << " " << fv->idx();
            j++ ;
        }   
        assert(j == 4);

        std::cerr << " f " << std::setw(4) << i
                  << " idx " << ssf.str()
                  << std::endl ;

    }   

}




  
int main()
{
  MyMesh mesh;

//#ifdef NEW_WAY
  // the request has to be called before a vertex/face/edge can be deleted. it grants access to the status attribute
  mesh.request_face_status();
  mesh.request_edge_status();
  mesh.request_vertex_status();
//#endif


  // generate vertices
  MyMesh::VertexHandle vh[8];
  MyMesh::FaceHandle   fh[6];

  vh[0] = mesh.add_vertex(MyMesh::Point(-1, -1,  1));
  vh[1] = mesh.add_vertex(MyMesh::Point( 1, -1,  1));
  vh[2] = mesh.add_vertex(MyMesh::Point( 1,  1,  1));
  vh[3] = mesh.add_vertex(MyMesh::Point(-1,  1,  1));
  vh[4] = mesh.add_vertex(MyMesh::Point(-1, -1, -1));
  vh[5] = mesh.add_vertex(MyMesh::Point( 1, -1, -1));
  vh[6] = mesh.add_vertex(MyMesh::Point( 1,  1, -1));
  vh[7] = mesh.add_vertex(MyMesh::Point(-1,  1, -1));

  dump(&mesh, "just vertices");

  std::vector<MyMesh::VertexHandle> tfv ; //  tmp_face_vhs;

  // generate (quadrilateral) faces
  tfv.clear(); tfv.push_back(vh[0]); tfv.push_back(vh[1]); tfv.push_back(vh[2]); tfv.push_back(vh[3]); fh[0] = mesh.add_face(tfv); 
  tfv.clear(); tfv.push_back(vh[7]); tfv.push_back(vh[6]); tfv.push_back(vh[5]); tfv.push_back(vh[4]); fh[1] = mesh.add_face(tfv); 
  tfv.clear(); tfv.push_back(vh[1]); tfv.push_back(vh[0]); tfv.push_back(vh[4]); tfv.push_back(vh[5]); fh[2] = mesh.add_face(tfv); 
  tfv.clear(); tfv.push_back(vh[2]); tfv.push_back(vh[1]); tfv.push_back(vh[5]); tfv.push_back(vh[6]); fh[3] = mesh.add_face(tfv); 
  tfv.clear(); tfv.push_back(vh[3]); tfv.push_back(vh[2]); tfv.push_back(vh[6]); tfv.push_back(vh[7]); fh[4] = mesh.add_face(tfv); 
  tfv.clear(); tfv.push_back(vh[0]); tfv.push_back(vh[3]); tfv.push_back(vh[7]); tfv.push_back(vh[4]); fh[5] = mesh.add_face(tfv);

  dump(&mesh, "after add_face*6");

   
  mesh.delete_face(fh[0], false);
  //  leave fh[1]   (vh[7], vh[6], vh[5], vh[4])
  mesh.delete_face(fh[2], false);
  mesh.delete_face(fh[3], false);
  mesh.delete_face(fh[4], false);
  mesh.delete_face(fh[5], false);
 
  mesh.garbage_collection();
  dump(&mesh, "after delete_face*5");


  // If isolated vertices result in a face deletion
  // they have to be deleted manually. If you want this
  // to happen automatically, change the second parameter
  // to true.
  // Now delete the isolated vertices 0, 1, 2 and 3

  mesh.delete_vertex(vh[0], false);
  mesh.delete_vertex(vh[1], false);
  mesh.delete_vertex(vh[2], false);
  mesh.delete_vertex(vh[3], false);

  mesh.garbage_collection();
  dump(&mesh, "after delete isolated");

  try {
        const char* path = "/tmp/DeleteFaceTest.off" ; 

        if ( !OpenMesh::IO::write_mesh(mesh, path) ) {
          std::cerr << "Cannot write mesh to file " << path << std::endl;
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
