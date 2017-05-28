//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <limits>
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
int NOpenMesh<T>::euler_characteristic()
{
    unsigned n_faces    = std::distance( mesh.faces_begin(),    mesh.faces_end() );
    unsigned n_vertices = std::distance( mesh.vertices_begin(), mesh.vertices_end() );
    unsigned n_edges    = std::distance( mesh.edges_begin(),    mesh.edges_end() );

    assert( n_faces    == mesh.n_faces() );
    assert( n_vertices == mesh.n_vertices() );
    assert( n_edges    == mesh.n_edges() );

    int euler = n_vertices - n_edges + n_faces  ;
    return euler ; 
}

template <typename T>
std::string NOpenMesh<T>::brief()
{
    std::stringstream ss ; 
    ss 
        << " V " << mesh.n_vertices()
        << " E " << mesh.n_edges()
        << " F " << mesh.n_faces()
        << " Euler [(V - E + F)] " << euler_characteristic() 
        ;
    return ss.str();
}


template <typename T>
bool NOpenMesh<T>::is_consistent_face_winding(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2)
{
    typedef typename T::HalfedgeHandle  HEH ; 
    typedef typename T::VertexHandle    VH ; 
   
    VH _vertex_handles[3] ; 
    _vertex_handles[0] = v0 ; 
    _vertex_handles[1] = v1 ; 
    _vertex_handles[2] = v2 ; 

    int i,ii,n(3) ; 

    struct WindingCheck  
    {
        HEH   halfedge_handle;
        bool is_new;
    };

    // checking based on PolyConnectivity::add_face

    std::vector<WindingCheck> edgeData_; 
    edgeData_.resize(n);

    for (i=0, ii=1; i<n; ++i, ++ii, ii%=n)
    {
        // Initialise edge attributes
        edgeData_[i].halfedge_handle = mesh.find_halfedge(_vertex_handles[i],
                                                          _vertex_handles[ii]);
        edgeData_[i].is_new = !edgeData_[i].halfedge_handle.is_valid();
  
        if (!edgeData_[i].is_new && !mesh.is_boundary(edgeData_[i].halfedge_handle))
        {
            std::cerr << "predicting... PolyMeshT::add_face: complex edge\n";
            return false ;
        }
    }
    return true ; 
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
             << " p " 
             << "[" 
             << std::setw(15) << std::fixed << std::setprecision(4) << p[0] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[1] << ","
             << std::setw(15) << std::fixed << std::setprecision(4) << p[2] << ","
             << "]"
             << " heh " << std::setw(5) << heh  
             ;

        if(heh_valid)
        {
            const VH& tvh = mesh.to_vertex_handle(heh);
            const VH& fvh = mesh.from_vertex_handle(heh);
            const FH& fh  = mesh.face_handle(heh);
            bool bnd = mesh.is_boundary(heh);

            std::cout  
                << " fvh->tvh " 
                << std::setw(3) << fvh << "->" 
                << std::setw(3) << tvh   
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
void NOpenMesh<T>::add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3, int verbosity )
{
   /*
              3-------2
              |     . | 
              |   .   |
              | .     |
              0-------1  
   */

    add_face_(v0,v1,v2, verbosity);
    add_face_(v2,v3,v0, verbosity);
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

    add_face_(vh[0],vh[1],vh[2],vh[3]);
    add_face_(vh[7],vh[6],vh[5],vh[4]);
    add_face_(vh[1],vh[0],vh[4],vh[5]);
    add_face_(vh[2],vh[1],vh[5],vh[6]);
    add_face_(vh[3],vh[2],vh[6],vh[7]);
    add_face_(vh[0],vh[3],vh[7],vh[4]);
}





template <typename T>
typename T::VertexHandle NOpenMesh<T>::find_vertex_exact(typename T::Point pt)  
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
typename T::VertexHandle NOpenMesh<T>::find_vertex_closest(typename T::Point pt, float& distance )  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 
    typedef typename T::VertexIter     VI ; 


    VI beg = mesh.vertices_begin() ;
    VI end = mesh.vertices_end() ;


    VH closest ;

    for (VI vit=beg ; vit != end ; ++vit) 
    {
        VH vh = *vit ; 
        const P& p = mesh.point(vh); 

        P d = p - pt ; 
        float dlen = d.length();

        if(dlen < distance )
        {
            closest = vh ;
            distance = dlen ;  
        }
    }
    return closest ; 
}


template <typename T>
typename T::VertexHandle NOpenMesh<T>::find_vertex_epsilon(typename T::Point pt, const float epsilon )  
{
    typedef typename T::VertexHandle   VH ;
    typedef typename T::Point           P ; 

    float distance = std::numeric_limits<float>::max() ;

    VH empty ; 
    VH closest = find_vertex_closest(pt, distance );
         
    return distance < epsilon ? closest : empty ;  
}


template <typename T>
typename T::VertexHandle NOpenMesh<T>::add_vertex_unique(typename T::Point pt, const float epsilon )  
{
    typedef typename T::VertexHandle VH ;

    VH prior = find_vertex_epsilon(pt, epsilon ) ;

    bool valid = mesh.is_valid_handle(prior) ;

    return valid ? prior : mesh.add_vertex(pt)   ; 
}


template <typename T>
typename T::FaceHandle NOpenMesh<T>::add_face_(typename T::VertexHandle v0, typename T::VertexHandle v1, typename T::VertexHandle v2, int verbosity )  
{
    typedef typename T::FaceHandle FH ; 

    assert( v0 != v1 );
    assert( v0 != v2 );
    assert( v1 != v2 );

    if(verbosity > 3)
    {
        std::cout << "add_face_"
                  << std::setw(5) << v0 
                  << std::setw(5) << v1 
                  << std::setw(5) << v2
                  << std::endl ;  

    }

    FH fh = mesh.add_face(v0,v1,v2);
    assert(mesh.is_valid_handle(fh));
    return fh ; 
}


template <typename T>
void NOpenMesh<T>::build_parametric(const nnode* node, int nu, int nv, int verbosity, const float epsilon)  
{
   /*

   Singularities like the poles of a latitude/longitude sphere parametrization 
   are handled by detecting the degenerate verts and adjusting the
   faces generated accordingly. 

   NB parameterizations should avoid leaning on the epsilon crutch, better 
   to provide exactly equal positions for poles and seams    


   */
    int ns = node->par_nsurf() ;
    auto vid = [ns,nu,nv](int s, int u, int v) { return  s*(nu+1)*(nv+1) + v*(nu+1) + u ; };

    int num_vert = (nu+1)*(nv+1)*ns ; 

    if(verbosity > 0)
    LOG(info) << "NOpenMesh<T>::build_parametric"
              << " ns " << ns
              << " nu " << nu
              << " nv " << nv
              << " num_vert(raw) " << num_vert 
              << " epsilon " << epsilon
              ;


    typedef typename T::VertexHandle VH ;
    typedef typename T::Point P ; 

    VH* vh = new VH[num_vert] ;

    int umin = 0 ; int umax = nu ; 
    int vmin = 0 ; int vmax = nv ; 

    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = vmin; v <= vmax ; v++)
        {
            for (int u = umin; u <= umax ; u++) 
            {
                nquad quv ; 
                quv.i.x  = u ; 
                quv.i.y  = v ; 
                quv.i.z  = nu ; 
                quv.i.w  = nv ; 

                glm::vec3 pos = node->par_pos(quv, s );

                vh[vid(s,u,v)] = add_vertex_unique(P(pos.x, pos.y, pos.z), epsilon) ;          
            }
        }
    }



    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = vmin; v < vmax; v++){
        for (int u = umin; u < umax; u++) 
        {
            int i00 = vid(s,u    ,     v) ;
            int i10 = vid(s,u + 1,     v) ;
            int i11 = vid(s,u + 1, v + 1) ;
            int i01 = vid(s,u    , v + 1) ;

            VH v00 = vh[i00] ;
            VH v10 = vh[i10] ;
            VH v01 = vh[i01] ;
            VH v11 = vh[i11] ;


            if(verbosity > 2)
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



         /*


            v
            ^
            4---5---6---7---8
            | / | \ | / | \ |
            3---4---5---6---7
            | \ | / | \ | / |
            2---3---4---5---6
            | / | \ | / | \ |
            1---2---3---4---5
            | \ | / | \ | / |
            0---1---2---3---4 > u      


            odd (u+v)%2 != 0 
           ~~~~~~~~~~~~~~~~~~~~

                  vmax
            (u,v+1)   (u+1,v+1)
              01---------11
               |       .  |
        umin   |  B  .    |   umax
               |   .      |
               | .   A    |
              00---------10
             (u,v)    (u+1,v)

                  vmin

            even   (u+v)%2 == 0 
            ~~~~~~~~~~~~~~~~~~~~~~

                   vmax
            (u,v+1)   (u+1,v+1)
              01---------11
               | .        |
       umin    |   .  D   |  umax
               |  C  .    |  
               |       .  |
              00---------10
             (u,v)    (u+1,v)
                   vmin

         */


            bool vmax_degenerate = v01 == v11 ; 
            bool vmin_degenerate = v00 == v10 ;

            bool umax_degenerate = v10 == v11 ; 
            bool umin_degenerate = v00 == v01 ;
  
            if( vmin_degenerate || vmax_degenerate ) assert( vmin_degenerate ^ vmax_degenerate ) ;
            if( umin_degenerate || umax_degenerate ) assert( umin_degenerate ^ umax_degenerate ) ;


            if( vmax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmax_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, verbosity );   // A (or C)
            } 
            else if ( vmin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmin_degenerate" << std::endl ; 
                add_face_( v11, v01, v10, verbosity  );  // D (or B)
            }
            else if ( umin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umin_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, verbosity );   // A (or D)
            }
            else if ( umax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umax_degenerate" << std::endl ; 
                add_face_( v00, v10, v01, verbosity ); // C  (or B)
            } 
            else if ((u + v) % 2)  // odd
            {
                add_face_( v00,v10,v11, verbosity  ); // A
                add_face_( v11,v01,v00, verbosity  ); // B
            } 
            else                 // even
            {
                add_face_( v00, v10, v01, verbosity  ); // C
                add_face_( v11, v01, v10, verbosity  ); // D
            }
        }
        }
    }


    int euler = euler_characteristic();
    int expect_euler = node->par_euler();
    bool euler_ok = euler == expect_euler ; 

    int nvertices = mesh.n_vertices() ;
    int expect_nvertices = node->par_nvertices(nu, nv);
    bool nvertices_ok = nvertices == expect_nvertices ; 

    if(verbosity > 0)
    {
        LOG(info) << brief() ; 
        LOG(info) << "build_parametric"
                  << " euler " << euler
                  << " expect_euler " << expect_euler
                  << ( euler_ok ? " EULER_OK " : " EULER_FAIL " )
                  << " nvertices " << nvertices
                  << " expect_nvertices " << expect_nvertices
                  << ( nvertices_ok ? " NVERTICES_OK " : " NVERTICES_FAIL " )
                  ;
        }

    if(!euler_ok || !nvertices_ok )
    {
        LOG(fatal) << "NOpenMesh::build_parametric : UNEXPECTED" ; 
        //dump("NOpenMesh::build_parametric : UNEXPECTED ");
    }

    //assert( euler_ok );
    //assert( nvertices_ok );
}








template struct NOpenMesh<NOpenMeshType> ;




