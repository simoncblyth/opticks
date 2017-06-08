
#include <boost/math/constants/constants.hpp>

#include "PLOG.hh"

#include "Nuv.hpp"
#include "NGLMext.hpp"
#include "NOpenMesh.hpp"
#include "NOpenMeshBuild.hpp"
#include "NNode.hpp"


template <typename T>
NOpenMeshBuild<T>::NOpenMeshBuild(
    T& mesh, 
    NOpenMeshProp<T>& prop, 
    const NOpenMeshDesc<T>& desc, 
    const NOpenMeshFind<T>& find,
    int verbosity
    )
    :
    mesh(mesh),
    prop(prop),
    desc(desc),
    find(find),
    verbosity(verbosity)
 {} 



template <typename T>
typename T::VertexHandle NOpenMeshBuild<T>::add_vertex_unique(typename T::Point pt, bool& added, const float epsilon )  
{
    VH prior = find.find_vertex_epsilon(pt, epsilon ) ;

    bool valid = mesh.is_valid_handle(prior) ;

    added = valid ? false : true  ;  

    return valid ? prior : mesh.add_vertex(pt)   ; 
}


template <typename T>
typename T::FaceHandle NOpenMeshBuild<T>::add_face_(typename T::VertexHandle v0, typename T::VertexHandle v1, typename T::VertexHandle v2, int identity, int permute )  
{
    FH f ; 
    if( v0 == v1 || v0 == v2 || v1 == v2 )
    {
        LOG(warning) << "add_face_ skipping degenerate "
                     << " " << std::setw(5) << v0
                     << " " << std::setw(5) << v1
                     << " " << std::setw(5) << v2 
                     ;
        return f ;  
    }

   

    VH u0, u1, u2 ; 

    if( permute == 0)
    {
        u0 = v0 ; 
        u1 = v1 ; 
        u2 = v2 ; 
    }
    else if( permute == 1)
    {
        u0 = v1 ; 
        u1 = v2 ; 
        u2 = v0 ; 
    }
    else if( permute == 2)
    {
        u0 = v2 ; 
        u1 = v0 ; 
        u2 = v1 ; 
    }


    if(verbosity > 3)
    {
        std::cout << "add_face_"
                  << " " << std::setw(5) << u0 
                  << " " << std::setw(5) << u1 
                  << " " << std::setw(5) << u2
                  << " id " << std::setw(5) << identity 
                  << std::endl ;  
    }

    f = mesh.add_face(u0,u1,u2);

    assert(mesh.is_valid_handle(f));

    if(identity > -1 ) prop.set_identity(f, identity ) ;

    return f ; 
}



template <typename T>
void NOpenMeshBuild<T>::add_face_(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2, typename T::VertexHandle v3 )
{
   /*
              3-------2
              |     . | 
              |   .   |
              | .     |
              0-------1  
   */

    add_face_(v0,v1,v2, -1);
    add_face_(v2,v3,v0, -1);
}


template <typename T>
bool NOpenMeshBuild<T>::is_consistent_face_winding(typename T::VertexHandle v0,typename T::VertexHandle v1, typename T::VertexHandle v2)
{
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
void NOpenMeshBuild<T>::add_parametric_primitive(const nnode* node, int level, int ctrl, float epsilon )  
{
   /*
   Singularities like the poles of a latitude/longitude sphere parametrization 
   are handled by detecting the degenerate verts and adjusting the
   faces generated accordingly. 

   NB parameterizations should avoid leaning on the epsilon crutch, better 
   to provide exactly equal positions for poles and seams    

   */

    int nu = 1 << level ; 
    int nv = 1 << level ; 

    int ns = node->par_nsurf() ;
    auto vid = [ns,nu,nv](int s, int u, int v) { return  s*(nu+1)*(nv+1) + v*(nu+1) + u ; };

    int num_vert = (nu+1)*(nv+1)*ns ; 

    if(verbosity > 0)
    LOG(info) << "NOpenMeshBuikd<T>::add_parametric_primitive"
              << " ns " << ns
              << " nu " << nu
              << " nv " << nv
              << " num_vert(raw) " << num_vert 
              << " epsilon " << epsilon
              << " ctrl " << ctrl 
              ;


    VH* vh = new VH[num_vert] ;

    int umin = 0 ; int umax = nu ; 
    int vmin = 0 ; int vmax = nv ; 

    for (int s=0 ; s < ns ; s++ )
    {
        for (int v = vmin; v <= vmax ; v++)
        {
            for (int u = umin; u <= umax ; u++) 
            {
                nuv uv = make_uv(s,u,v,nu,nv);

                glm::vec3 pos = node->par_pos(uv);

                bool added(false) ;

                int vidx = vid(s,u,v) ;

                vh[vidx] = add_vertex_unique(P(pos.x, pos.y, pos.z), added, epsilon) ; 

                if(added)
                {
                    //mesh.property(prop.v_parametric, vh[vidx] ) = uv   ;
                    prop.set_uv( vh[vidx], uv );
                } 
         
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


            int identity = -1 ; 

            if( vmax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmax_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, identity );   // A (or C)
            } 
            else if ( vmin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "vmin_degenerate" << std::endl ; 
                add_face_( v11, v01, v10, identity  );  // D (or B)
            }
            else if ( umin_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umin_degenerate" << std::endl ; 
                add_face_( v00,v10,v11, identity );   // A (or D)
            }
            else if ( umax_degenerate )
            {
                if(verbosity > 2)
                std::cout << "umax_degenerate" << std::endl ; 
                add_face_( v00, v10, v01, identity ); // C  (or B)
            } 
            else if ((u + v) % 2)  // odd
            {
                add_face_( v00,v10,v11, identity  ); // A
                add_face_( v11,v01,v00, identity  ); // B
            } 
            else                 // even
            {
                add_face_( v00, v10, v01, identity  ); // C
                add_face_( v11, v01, v10, identity  ); // D
            }
        }
        }
    }
}


template <typename T>
void NOpenMeshBuild<T>::euler_check(const nnode* node, int level )
{
    int nu = 1 << level ; 
    int nv = 1 << level ; 

    int euler = desc.euler_characteristic();
    int expect_euler = node->par_euler();
    bool euler_ok = euler == expect_euler ; 

    int nvertices = mesh.n_vertices() ;
    int expect_nvertices = node->par_nvertices(nu, nv);
    bool nvertices_ok = nvertices == expect_nvertices ; 

    if(verbosity > 0)
    {
        LOG(info) << desc.desc_euler() ; 
        LOG(info) << "euler_check"
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
        LOG(fatal) << "NOpenMeshBuild::euler_check : UNEXPECTED" ; 
        //dump("NOpenMesh::build_parametric : UNEXPECTED ");
    }

    //assert( euler_ok );
    //assert( nvertices_ok );
}







template <typename T>
void NOpenMeshBuild<T>::mark_faces(const nnode* other)
{
    typedef typename T::FaceIter            FI ; 

    for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        mark_face( fh, other );
    }
    if(verbosity > 0)
    LOG(info) << "mark_faces " <<  desc_facemask() ;  
}
 

template <typename T>
void NOpenMeshBuild<T>::mark_face(typename T::FaceHandle fh, const nnode* other)
{

/*
facemask "f_inside_outside"
-----------------------------

bitmask of 3 bits for each face corresponding to 
inside/outside for the vertices of the face

0   (0) : all vertices outside other

1   (1) : 1st vertex inside other
2  (10) : 2nd vertex inside other
3  (11) : 1st and 2nd vertices inside other
4 (100) : 3rd vertex inside other
5 (101) : 1st and 3rd vertex inside other
6 (110) : 2nd and 3rd vertex inside other

7 (111) : all vertices inside other 

*/

    std::function<float(float,float,float)> sdf = other->sdf();

    typedef typename T::ConstFaceVertexIter FVI ; 

    int _facemask = 0 ;  

    assert( mesh.valence(fh) == 3 );

    int fvert = 0 ; 

    for(FVI fv=mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
    {
        const VH vh = *fv ; 
    
        const P pt = mesh.point(vh);

        float dist = sdf(pt[0], pt[1], pt[2]);

        mesh.property(prop.v_sdf_other, vh ) = dist   ;

        bool inside_other = dist < 0.f ; 

        _facemask |=  (!!inside_other << fvert++) ;   
    }
    assert( fvert == 3 );

    //mesh.property(prop.f_inside_other, fh) = _facemask ; 
    prop.set_facemask(fh, _facemask );

    if(f_inside_other_count.count(_facemask) == 0)
    {
        f_inside_other_count[_facemask] = 0 ; 
    }
    f_inside_other_count[_facemask]++ ;

 
}
 

template <typename T>
std::string NOpenMeshBuild<T>::desc_facemask()
{
    std::stringstream ss ; 
    typedef std::map<int,int> MII ; 

    for(MII::const_iterator it=f_inside_other_count.begin() ; it!=f_inside_other_count.end() ; it++)
    {
         ss << std::setw(1) << it->first << ":" << std::setw(6) << it->second << "|" ; 
    }

    return ss.str();
}
 


template <typename T>
void NOpenMeshBuild<T>::copy_faces(const NTriSource*   other, float epsilon  )
{
    unsigned nvtx = other->get_num_vert();
    unsigned ntri = other->get_num_tri();

    glm::uvec3 tri ; 
    glm::vec3 pos ; 
    //glm::vec3 nrm ; 
    //glm::vec3 uv ;
 
    std::vector<VH> vhs ; 
    vhs.reserve(nvtx);

    for(unsigned i=0 ; i < nvtx ; i++)
    {
        other->get_vert(i, pos );
        //other->get_normal(i, nrm );
        //other->get_uv(i, uv );

         bool added(false);        
         VH vh = add_vertex_unique(P(pos.x, pos.y, pos.z), added, epsilon);  
         vhs.push_back(vh); 
    }

    unsigned badface(0);
    for(unsigned i=0 ; i < ntri ; i++)
    {
        other->get_tri(i, tri );

        int identity = -1 ; 
        FH f = add_face_( vhs[tri.x], vhs[tri.y], vhs[tri.z], identity ); 
        if(!mesh.is_valid_handle(f)) badface++ ; 
    }
    assert(badface == 0);
}

 
template <typename T>
void NOpenMeshBuild<T>::copy_faces(const NOpenMesh<T>* other, int facemask, float epsilon )
{
    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 


    unsigned badface(0);

    for( FI f=other->mesh.faces_begin() ; f != other->mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        int _f_inside_other = other->mesh.property(other->prop.f_inside_other, fh) ; 

        if( _f_inside_other == facemask )
        {  
            VH nvh[3] ; 
            int fvert(0);
            for(FVI fv=other->mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
            {
                const VH& vh = *fv ; 
                const P& pt = other->mesh.point(vh);
                bool added(false);        
                nvh[fvert++] = add_vertex_unique(P(pt[0], pt[1], pt[2]), added, epsilon);  
            } 
            assert( fvert == 3 );

            int identity = -1 ; 
            FH f = add_face_( nvh[0], nvh[1], nvh[2], identity ); 
            if(!mesh.is_valid_handle(f)) badface++ ; 

        }
    }

    if(badface > 0)
    {
        LOG(warning) << "copy_faces badface skips: " << badface ; 
    }
}
 






template <typename T>
void NOpenMeshBuild<T>::add_tripatch()
{
    if(verbosity > 1) LOG(info) << "add_tripatch" ; 

    double pi = boost::math::constants::pi<double>() ;
    double sa[2],ca[2] ;
    for(unsigned i=0 ; i < 2 ; i++) sincos_<double>( 2.*pi*i/6.0, sa[i], ca[i] ); 

    float s = 200.f ; 
    float z = 0.f ; 

    P p[6] ; 
    VH v[6];

    p[0] = P(       0,       0,  z);
    p[1] = P( s*ca[0], s*sa[0], z );
    p[2] = P( s*ca[1], s*sa[1], z );
    p[3] = P( 2.*s*ca[0], 2*s*sa[0], z );
    p[5] = P( 2.*s*ca[1], 2*s*sa[1], z );
    p[4] = P( s*(1.+ca[1]),  s*sa[1], z );

    P mid = p[1] + p[2] + p[4] ;
    mid /= 3.0 ; 

    for(unsigned i=0 ; i < 6 ; i++) v[i] = mesh.add_vertex(p[i] - mid) ;

    // check suspicion of vertex ordering dependancy   
    // seems to not be so, so long as face order is OK, see below
    //int permute[4] = {2,0,1,1} ;
    //int permute[4] = {1,2,0,1} ;
    int permute[4] = {0,0,0,0} ;

    add_face_(v[1],v[4],v[2], 1, permute[0] );
    add_face_(v[1],v[2],v[0], 2, permute[1] );
    add_face_(v[4],v[1],v[3], 3, permute[2] );
    add_face_(v[2],v[4],v[5], 4, permute[3] );

    // add_face_ order dictates subdiv order...
    // see NOpenMeshSubdiv.cpp for notes about face ordering sensitivity of sqrt3 subdiv
}

template <typename T>
const char* NOpenMeshBuild<T>::TRIPATCH = R"LITERAL("
    /*
    6  verts :                                 4 faces
                                               3 corner faces with 2 boundary edges 
                                                                     
                 5                                +           
                / \                              /4\
               2---4                            +---+
              / \ / \                          /2\1/3\
             0---1---3                        +---+---+
    */
)LITERAL" ;





template <typename T>
void NOpenMeshBuild<T>::add_hexpatch(bool inner_only)
{
    if(verbosity > 1) LOG(info) << "add_hexpatch" ; 
 
    float s = 200.f ; 
    float z = 0.f ; 

    double pi = boost::math::constants::pi<double>() ;
    double sa[7],ca[7] ;
    for(unsigned i=0 ; i < 7 ; i++) sincos_<double>( 2.*pi*i/6.0, sa[i], ca[i] ); 


    P p[19] ; 
    VH v[19];

    enum { A=10, B=11, C=12, D=13, E=14, F=15, G=16, H=17, I=18, J=19, K=20, L=21, M=22, N=23, O=24 } ; 

    p[0] = P( 0,  0,  z);

    for(int i=1 ; i <= 6 ; i++)   p[i] = P(   s*ca[i-1],   s*sa[i-1],  z);
    for(int i=1 ; i <= 6 ; i++) p[6+i] = P( 2*s*ca[i-1], 2*s*sa[i-1],  z);

    p[D] = P(  s*(1+ca[1]),     s*sa[1],z) ; 
    p[F] = P( -s*(1+ca[1]),     s*sa[1],z) ; 
    p[G] = P( -s*(1+ca[1]),    -s*sa[1],z) ; 
    p[I] = P(  s*(1+ca[1]),    -s*sa[1],z) ; 

    p[E] = P(            0,  2.*s*sa[1],z) ; 
    p[H] = P(            0, -2.*s*sa[1],z) ; 

    unsigned vlast = inner_only ? 6 : I ; 

    for(unsigned i=0 ; i <=vlast ; i++) v[i] = mesh.add_vertex(p[i]) ;

    add_face_(v[0],v[1],v[2], 1);
    add_face_(v[0],v[2],v[3], 2);
    add_face_(v[0],v[3],v[4], 3);
    add_face_(v[0],v[4],v[5], 4);
    add_face_(v[0],v[5],v[6], 5);
    add_face_(v[0],v[6],v[1], 6);

    if(!inner_only)
    {
        add_face_(v[1],v[I],v[7], O);  // <-- huh with this at end, get a missed edge with 1st round of boundary face subdiv
        add_face_(v[1],v[7],v[D], 7);
        add_face_(v[1],v[D],v[2], 8);

        add_face_(v[2],v[D],v[8], 9);
        add_face_(v[2],v[8],v[E], A);
        add_face_(v[2],v[E],v[3], B);

        add_face_(v[3],v[E],v[9], C);
        add_face_(v[3],v[9],v[F], D);
        add_face_(v[3],v[F],v[4], E);

        add_face_(v[4],v[F],v[A], F);
        add_face_(v[4],v[A],v[G], G);
        add_face_(v[4],v[G],v[5], H);

        add_face_(v[5],v[G],v[B], I);
        add_face_(v[5],v[B],v[H], J);
        add_face_(v[5],v[H],v[6], K);

        add_face_(v[6],v[H],v[C], L);
        add_face_(v[6],v[C],v[I], M);
        add_face_(v[6],v[I],v[1], N);
    }
}


template <typename T>
const char* NOpenMeshBuild<T>::HEXPATCH = R"LITERAL("
/*
    19 verts :                                 24 faces
    12 around boundary, 7 in interior          12 with edge on boundary
                                               (no corners ie 2 boundary edges)
                                                                     
                 9---e---8                        +---+---+          
                / \ / \ / \                      /d\c/b\a/9\
               f---3---2---d                    +---+---+---+        
              / \ / \ / \ / \                  /f\e/3\2/1\8/7\
             a---4---0---1---7                +---+---+---+---+      
              \ / \ / \ / \ /                  \g/h\4/5\6/n\o/       
               g---5---6---i                    +---5---6---i        
                \ / \ / \ /                      \i/j\k/l\m/         
                 b---h---c                        +---+---+          
*/
)LITERAL" ;




// suspect need to use "rosette" pattern of minimum seed vertices
// to fit in with sqrt3_subdiv_r
//
// add_face_(v[E],v[9],v[3], C);
// add_face_(v[9],v[F],v[3], D); 
//  
// sqrt3_split_r is sensitive to the vertex ordering!!
// above (which is a valid winding) leads to a missed edge  
// adopting an ordering more consistent with the other faces
// solve the issue
//
//
//
/*                                                                 
                 +-------+        
                / \     / \
               /   \ 2 /   \
              /  3  \ /  1  \
             +-------+-------+   
              \  4  / \  6  /   
               \   / 5 \   /   
                \ /     \ /   
                 +-------+   

*/             




template <typename T>
void NOpenMeshBuild<T>::add_tetrahedron()
{
    if(verbosity > 1) LOG(info) << "add_tetrahedron" ; 


    /*
     https://en.wikipedia.org/wiki/Tetrahedron

       (1,1,1), (1,−1,−1), (−1,1,−1), (−1,−1,1)



                 +-----------0
                /|          /| 
               / |         / |
              3-----------+  |
              |  |        |  |
              |  |        |  |
              |  2--------|--+
              | /         | /
              |/          |/
              +-----------1
          

         z  y
         | /
         |/
         +---> x

    */

    typedef typename T::Point P ; 
    typename T::VertexHandle vh[4];

    float s = 500.f ;  

    vh[0] = mesh.add_vertex(P( s,  s,  s));
    vh[1] = mesh.add_vertex(P( s, -s, -s));
    vh[2] = mesh.add_vertex(P(-s,  s, -s));
    vh[3] = mesh.add_vertex(P(-s, -s,  s));

    int identity = -1 ;  
    add_face_(vh[0],vh[3],vh[1],identity);
    add_face_(vh[3],vh[2],vh[1],identity);
    add_face_(vh[0],vh[2],vh[3],identity);
    add_face_(vh[2],vh[0],vh[1],identity);
}



template <typename T>
void NOpenMeshBuild<T>::add_cube()
{
    if(verbosity > 1) LOG(info) << "add_cube" ; 
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





template struct NOpenMeshBuild<NOpenMeshType> ;

