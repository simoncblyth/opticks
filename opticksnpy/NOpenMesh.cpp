//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <iostream>
#include <sstream>

#include "PLOG.hh"
#include "NGLM.hpp"
#include "NNode.hpp"
#include "NOpenMesh.hpp"

template <typename T>
int NOpenMesh<T>::write(const char* path)
{
    try
    {
      if ( !OpenMesh::IO::write_mesh(mesh, path) )
      {
        std::cerr << "Cannot write mesh to file " << path << std::endl;
        return 1;
      }
    }
    catch( std::exception& x )
    {
      std::cerr << x.what() << std::endl;
      return 1;
    }
    return 0 ; 
}


template <typename T>
void NOpenMesh<T>::dump(const char* msg)
{
    LOG(info) << msg << " " << brief() ; 
    dump_vertices();
    dump_faces();
}


template <typename T>
std::string NOpenMesh<T>::brief()
{
    std::stringstream ss ; 

    unsigned nface = std::distance( mesh.faces_begin(), mesh.faces_end() );
    unsigned nvert = std::distance( mesh.vertices_begin(), mesh.vertices_end() );
    unsigned nedge = std::distance( mesh.edges_begin(), mesh.edges_end() );

    unsigned n_face = mesh.n_faces(); 
    unsigned n_vert = mesh.n_vertices(); 
    unsigned n_edge = mesh.n_edges();

    assert( nface == n_face );
    assert( nvert == n_vert );
    assert( nedge == n_edge );

    ss 
        << " V " << nvert
        << " F " << nface
        << " E " << nedge
        << " (V - E + F) - 2  (expect 0) " << nvert - nedge + nface - 2
        ;
    
    return ss.str();
}


template <typename T>
void NOpenMesh<T>::dump_vertices(const char* msg)
{
    LOG(info) << msg ; 

    typename T::VertexIter v_it, v_end(mesh.vertices_end());

    for (v_it=mesh.vertices_begin(); v_it!=v_end ; ++v_it) std::cout << mesh.point( *v_it ) << std::endl ;
}


template <typename T>
void NOpenMesh<T>::dump_faces(const char* msg )
{
    LOG(info) << msg << " nface " << mesh.n_faces() ; 

    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::Point               P ; 

    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        int f_idx = f->idx() ;  
        std::cout << " f " << std::setw(4) << *f 
                  << " i " << std::setw(3) << f_idx 
                  << " v " << std::setw(3) << mesh.valence(*f) 
                  << " : " 
                  ; 

        // over points of the face 
        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout << std::setw(3) << *fv << " " ;

        for(FVI fv=mesh.cfv_iter(*f) ; fv.is_valid() ; fv++) 
             std::cout 
                       << std::setprecision(3) << std::fixed << std::setw(20) 
                       << mesh.point(*fv) << " "
                       ;

        std::cout << std::endl ; 
    }
}
 
template <typename T>
void NOpenMesh<T>::add_face(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3 )
{
    std::vector<typename T::VertexHandle>  face_vhandles;

    face_vhandles.push_back(v0);
    face_vhandles.push_back(v1);
    face_vhandles.push_back(v2);
    face_vhandles.push_back(v3);

    mesh.add_face(face_vhandles);
}
 
template <typename T>
void NOpenMesh<T>::build_cube()
{
    /*

                 3-----------2
                /|          /| 
               / |         / |
              0-----------1  |
              |  |        |  |
              |  |        |  |
              |  7--------|--6
              | /         | /
              |/          |/
              4-----------5
          

         z  y
         | /
         |/
         +---> x

    */

    typedef typename T::Point P ; 
    typename T::VertexHandle vh[8];

    vh[0] = mesh.add_vertex(P(-1, -1,  1));
    vh[1] = mesh.add_vertex(P( 1, -1,  1));
    vh[2] = mesh.add_vertex(P( 1,  1,  1));
    vh[3] = mesh.add_vertex(P(-1,  1,  1));
    vh[4] = mesh.add_vertex(P(-1, -1, -1));
    vh[5] = mesh.add_vertex(P( 1, -1, -1));
    vh[6] = mesh.add_vertex(P( 1,  1, -1));
    vh[7] = mesh.add_vertex(P(-1,  1, -1));

    add_face(vh[0],vh[1],vh[2],vh[3]);
    add_face(vh[7],vh[6],vh[5],vh[4]);
    add_face(vh[1],vh[0],vh[4],vh[5]);
    add_face(vh[2],vh[1],vh[5],vh[6]);
    add_face(vh[3],vh[2],vh[6],vh[7]);
    add_face(vh[0],vh[3],vh[7],vh[4]);
}



template <typename T>
void NOpenMesh<T>::build_parametric(const nnode* node, int usteps, int vsteps)  
{
    int num_vert = (usteps+1)*(vsteps+1) ; 
    //int num_tri = usteps*vsteps*2 ; 

    typedef typename T::VertexHandle VH ;
    typedef typename T::Point P ; 

    VH* vh = new VH[num_vert] ;

    auto vid = [usteps](int u, int v) { return v * (usteps + 1) + u ; };

    for (int v = 0; v <= vsteps; v++){
    for (int u = 0; u <= usteps; u++) 
    {
        glm::vec2 uv( float(u)/(float)usteps, float(v)/(float)vsteps );
        glm::vec3 pos = node->par_pos(uv);

        vh[vid(u,v)] = mesh.add_vertex(P(pos.x, pos.y, pos.z)) ;          
    }
    }


    for (int v = 0; v < vsteps; v++){
    for (int u = 0; u < usteps; u++) 
    {
        if ((u + v) % 2) 
        {
            mesh.add_face( 
                           vh[vid(u, v)], 
                           vh[vid(u + 1, v)], 
                           vh[vid(u + 1, v + 1)]
                      );

            mesh.add_face( 
                           vh[vid(u + 1, v + 1)], 
                           vh[vid(u    , v + 1)], 
                           vh[vid(u    , v    )]
                         );
        } 
        else 
        {
            mesh.add_face( 
                           vh[vid(u, v)], 
                           vh[vid(u + 1, v)], 
                           vh[vid(u    , v + 1)]
                      );

            mesh.add_face( 
                           vh[vid(u + 1, v + 1)], 
                           vh[vid(u    , v + 1)], 
                           vh[vid(u + 1, v    )]
                         );
 
        }
    }
    }


}





template struct NOpenMesh<NOpenMeshType> ;




