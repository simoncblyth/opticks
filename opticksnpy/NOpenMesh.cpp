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

    unsigned nface = std::distance( mesh.faces_begin(),    mesh.faces_end() );
    unsigned nvert = std::distance( mesh.vertices_begin(), mesh.vertices_end() );
    unsigned nedge = std::distance( mesh.edges_begin(),    mesh.edges_end() );

    unsigned n_face = mesh.n_faces(); 
    unsigned n_vert = mesh.n_vertices(); 
    unsigned n_edge = mesh.n_edges();

    assert( nface == n_face );
    assert( nvert == n_vert );
    assert( nedge == n_edge );

    int euler = nvert - nedge + nface  ;

    ss 
        << " V " << nvert
        << " F " << nface
        << " E " << nedge
        << " euler [(V - E + F)]  (expect 2) " << euler
        ;
    
    return ss.str();
}


template <typename T>
void NOpenMesh<T>::dump_vertices(const char* msg)
{
    LOG(info) << msg ; 

    typedef typename T::Point          P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::HalfedgeHandle HEH ; 
    typedef typename T::FaceHandle     FH ; 
    typedef typename T::Vertex         V ; 
    typedef typename T::VertexIter     VI ; 

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {

        VH vh = *vit ; 
        int idx = vh.idx() ;
        assert( idx == std::distance( beg, vit ) ) ;

        const P& p = mesh.point(vh); 

        //const V& v = mesh.vertex(vh); // <-- purely private internal V has no public methods, so useless

        const HEH& heh = mesh.halfedge_handle(vh); 
        bool heh_valid = mesh.is_valid_handle(heh);


        std::cout 
             << " vh " << std::setw(5) << vh  
             << " p " << p
             << " heh " << std::setw(5) << heh  
             ;

        if(heh_valid)
        {
            const VH& tvh = mesh.to_vertex_handle(heh);
            const VH& fvh = mesh.from_vertex_handle(heh);
            const FH& fh  = mesh.face_handle(heh);
            bool bnd = mesh.is_boundary(heh);

            std::cout  
                << " tvh " << std::setw(5) << tvh  
                << " fvh " << std::setw(5) << fvh  
                << " fh " << std::setw(5) << fh  
                << " bnd " << std::setw(5) << bnd 
                << std::endl ;
        }
        else
        {
             std::cout << std::endl ; 
        }

    }
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
   /*
              3-------2
              |     . | 
              |   .   |
              | .     |
              0-------1  
   */

    mesh.add_face(v0,v1,v2);
    mesh.add_face(v2,v3,v0);
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
typename T::VertexHandle NOpenMesh<T>::find_vertex(typename T::Point pt)  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 
    typedef typename T::VertexIter     VI ; 

    VH result ;

    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 
        if( p == pt )
        {
            result = vh ; 
            break ;          
        }
    }
    return result ; 
}


template <typename T>
typename T::VertexHandle NOpenMesh<T>::add_vertex_unique(typename T::Point pt)  
{
    typedef typename T::VertexHandle VH ;
    VH prior = find_vertex(pt);
    bool valid = mesh.is_valid_handle(prior) ;
    return valid ? prior : mesh.add_vertex(pt)   ; 
}


template <typename T>
void NOpenMesh<T>::build_parametric(const nnode* node, int nu, int nv)  
{
   /*
    suspect that cannot generically do this, eg consider the sphere
    need to handle the poles different from the rest ...

   */
    int ns = node->par_nsurf() ;
    auto vid = [ns,nu,nv](int s, int u, int v) { return  s*(nu+1)*(nv+1) + v*(nu+1) + u ; };

    int num_vert = (nu+1)*(nv+1)*ns ; 

    typedef typename T::VertexHandle VH ;
    typedef typename T::Point P ; 

    VH* vh = new VH[num_vert] ;

    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = 0; v <= nv ; v++){
        for (int u = 0; u <= nu ; u++) 
        {
            glm::vec2 uv( float(u)/nu, float(v)/nv );
            glm::vec3 pos = node->par_pos(uv, s );

            vh[vid(s,u,v)] = add_vertex_unique(P(pos.x, pos.y, pos.z)) ;          
        }
        }
    }


    /*
        4---5---6---7---8
        | \ | / | \ | / |
        3---4---5---6---7
        | / | \ | / | \ |
        2---3---4---5---6
        | \ | / | \ | / |
        1---2---3---4---5
        | / | \ | / | \ |
        0---1---2---3---4    -> u      

    */

    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = 0; v < nv; v++){
        for (int u = 0; u < nu; u++) 
        {
            int i00 = vid(s,u    ,     v) ;
            int i10 = vid(s,u + 1,     v) ;
            int i11 = vid(s,u + 1, v + 1) ;
            int i01 = vid(s,u    , v + 1) ;

            VH v00 = vh[i00] ;
            VH v10 = vh[i10] ;
            VH v01 = vh[i01] ;
            VH v11 = vh[i11] ;

            std::cout 
                  << " s " << std::setw(3)  << s
                  << " v " << std::setw(3)  << v
                  << " u " << std::setw(3)  << u
                  << " v00 " << std::setw(3)  << v00
                  << " v10 " << std::setw(3)  << v10
                  << " v01 " << std::setw(3)  << v01
                  << " v11 " << std::setw(3)  << v11
                  << std::endl 
                  ;
   

            if ((u + v) % 2) 
            {
         /*
            (u,v+1)   (u+1,v+1)
              01---------11
               |       .  |
               |     .    |
               |   .      |
               | .        |
              00---------10
             (u,v)    (u+1,v)

         */

                mesh.add_face( v00,v10,v11 );
                mesh.add_face( v11,v01,v00 );
            } 
            else 
            {
         /*

            (u,v+1)   (u+1,v+1)
              01---------11
               | .        |
               |   .      |
               |     .    |
               |       .  |
              00---------10
             (u,v)    (u+1,v)

         */
                mesh.add_face( v00, v10, v01 );
                mesh.add_face( v11, v01, v10 );
            }
        }
        }
    }

}





template struct NOpenMesh<NOpenMeshType> ;




