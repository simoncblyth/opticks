//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <limits>
#include <iostream>
#include <sstream>


#include "PLOG.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"
#include "NNode.hpp"

#include "NOpenMesh.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"

#include <OpenMesh/Tools/Utils/MeshCheckerT.hh>


template <typename T>
NOpenMesh<T>::NOpenMesh(const nnode* node, int level, int verbosity, int ctrl, float epsilon)
    :
    prop(mesh),
    desc(mesh, prop),
    find(mesh, prop),
    build(mesh, prop, desc, find),
    subdiv(mesh, prop, desc, find, build),

    node(node), 
    level(level), 
    verbosity(verbosity), 
    ctrl(ctrl), 
    epsilon(epsilon),
    nsubdiv(1),
    leftmesh(NULL),
    rightmesh(NULL)
{
    init();
}


template <typename T>
void NOpenMesh<T>::init()
{
    if(node)
    {
        build_parametric(); 
        check();
    }
}

template <typename T>
void NOpenMesh<T>::check()
{
    assert(OpenMesh::Utils::MeshCheckerT<T>(mesh).check()) ;
    LOG(info) << "NOpenMesh<T>::check OK" ; 
}


template <typename T>
void NOpenMesh<T>::subdiv_test()
{
    typedef typename T::FaceHandle          FH ; 

    unsigned nloop0 = find.find_boundary_loops() ;

    std::vector<FH> target_faces ; 

    //unsigned param = 0 ;  // margin for INTERIOR_FACE
    unsigned param = 6 ;  // valence for REGULAR_FACE
 
    //find.find_faces( target_faces, FIND_REGULAR_FACE,  param );
    find.find_faces( target_faces, FIND_ALL_FACE,  param );


    unsigned n_target_faces = target_faces.size() ;

    LOG(info) << "subdiv_interior_test" 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " n_target_faces " << n_target_faces
              << " nloop0 " << nloop0
              ;


    if( n_target_faces > 0)
    {
        for(unsigned i=0 ; i < n_target_faces ; i++) 
        {
            FH fh = target_faces[i] ;
            subdiv.manual_subdivide_face(fh,  NULL, verbosity, epsilon);
        }
        // subdiv.refine(fh);  <-- just hangs ? infinite loop ?
    }

    unsigned nloop1 = find.find_boundary_loops() ;

    LOG(info) << "subdiv_test DONE " 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " nloop1 " << nloop1
              ;
}





template <typename T>
void NOpenMesh<T>::build_parametric()
{
/*

Aim is to combine leftmesh and rightmesh faces 
appropriately for the CSG operator of the combination. 

Faces that are entirely inside or outside the other 
can simply be copied or not copied as appropriate into the combined mesh.  

The complex part is how to stitch up the join between the two(? or more) open boundaries. 

union(A,B)
    copy faces of A entirely outside B and vv  (symmetric/commutative)

intersection(A,B)
    copy faces of A entirely inside B and vv (symmetric/commutative)

difference(A,B) = intersection(A,-B)      (asymmetric/not-commutative)
    copy faces of A entirely outside -B 
    copy faces of A entirely inside B 

    copy faces of A entirely outside B 
    copy faces of B entirely inside A 

*/

    bool combination = node->left && node->right ; 

    if(!combination)
    {
        build.add_parametric_primitive(node, level, verbosity, ctrl, epsilon ); // adds unique vertices and faces to build out the parametric mesh  
        build.euler_check(node, level, verbosity );
    }
    else
    {
        assert(node->type == CSG_UNION || node->type == CSG_INTERSECTION || node->type == CSG_DIFFERENCE );
   
        typedef NOpenMesh<T> Mesh ; 
        leftmesh = new Mesh(node->left, level, verbosity, epsilon) ; 
        rightmesh = new Mesh(node->right, level, verbosity, epsilon) ; 

        LOG(info) << "build_parametric" 
                  << " leftmesh " << leftmesh
                  << " rightmesh " << rightmesh
                   ;

        // initial meshes should be closed with no boundary loops
        assert( leftmesh->find.find_boundary_loops() == 0) ;   
        assert( rightmesh->find.find_boundary_loops() == 0) ;  

        leftmesh->build.mark_faces( node->right );
        LOG(info) << "[0] leftmesh inside node->right : " <<  leftmesh->build.desc_inside_other() ;  

        rightmesh->build.mark_faces( node->left );
        LOG(info) << "[0] rightmesh inside node->left : " <<  rightmesh->build.desc_inside_other() ;  

        /**

        0. sub-object mesh tris are assigned a facemask property from (0-7) (000b-111b) 
           indicating whether vertices have negative other sub-object SDF values  

        1. CSG sub-object mesh faces with mixed other sub-object sdf signs (aka border faces) 
           are subdivided (ie original tris are deleted and replaced with four new smaller ones)

        2. border subdiv is repeated in nsubdiv rounds, increasing mesh "resolution" around the border
        
        3. Only triangles with all 3 vertices inside or outside the other sub-object get copied
           into the composite mesh, this means there is no overlapping/intersection at this stage  

        4. remaining gap between sub-object meshes then needs to be zippered up 
           to close the composite mesh ???? how exactly 
     
        **/

       

        leftmesh->subdivide_border_faces( node->right, nsubdiv );
        LOG(info) << "[1] leftmesh inside node->right : " <<  leftmesh->build.desc_inside_other() ;  

        rightmesh->subdivide_border_faces( node->left, nsubdiv );
        LOG(info) << "[1] rightmesh inside node->left : " <<  rightmesh->build.desc_inside_other() ;  


        leftmesh->mesh.garbage_collection();  
        rightmesh->mesh.garbage_collection();  


        // despite subdiv should still be closed zero boundary meshes
        //  BUT WAS GETTING LOADS OF OPEN LOOPS, when using _creating_soup subdiv approach

        assert( leftmesh->find.find_boundary_loops() == 0) ;   
        assert( rightmesh->find.find_boundary_loops() == 0) ;  


        if(node->type == CSG_UNION)
        {
            build.copy_faces( leftmesh,  ALL_OUTSIDE_OTHER, verbosity, epsilon );
            build.copy_faces( rightmesh, ALL_OUTSIDE_OTHER, verbosity, epsilon );
        }
        else if(node->type == CSG_INTERSECTION)
        {
            build.copy_faces( leftmesh,  ALL_INSIDE_OTHER, verbosity, epsilon  );
            build.copy_faces( rightmesh, ALL_INSIDE_OTHER, verbosity, epsilon  );
        }
        else if(node->type == CSG_DIFFERENCE )
        {
            build.copy_faces( leftmesh,  ALL_OUTSIDE_OTHER, verbosity, epsilon  );
            build.copy_faces( rightmesh, ALL_INSIDE_OTHER, verbosity, epsilon  );
        }
    }
}


template <typename T>
bool NOpenMesh<T>::is_border_face(const int facemask)
{
    return !( facemask == ALL_OUTSIDE_OTHER || facemask == ALL_INSIDE_OTHER ) ; 
}

template <typename T>
void NOpenMesh<T>::subdivide_border_faces(const nnode* other, unsigned nsubdiv, bool creating_soup )
{

    typedef typename T::FaceHandle          FH ; 
    typedef typename T::FaceIter            FI ; 

    std::vector<FH> border_faces ; 

    for(unsigned round=0 ; round < nsubdiv ; round++)
    {
        border_faces.clear();
        for( FI f=mesh.faces_begin() ; f != mesh.faces_end(); ++f ) 
        {
            FH fh = *f ;  
            int _f_inside_other = mesh.property(prop.f_inside_other, fh) ; 
            if(!is_border_face(_f_inside_other)) continue ; 
            border_faces.push_back(fh);
        }

        LOG(info) << "subdivide_border_faces" 
                  << " nsubdiv " << nsubdiv
                  << " round " << round 
                  << " nbf: " << border_faces.size()
                  << " creating_soup " << creating_soup
                  ;


        for(unsigned i=0 ; i < border_faces.size(); i++) 
        {
            FH fh = border_faces[i] ;
            if(creating_soup)
            {
                subdiv.manual_subdivide_face_creating_soup(fh, other, verbosity, epsilon );
            }
            else
            {        
                subdiv.manual_subdivide_face(fh, other, verbosity, epsilon); 
            }
        }
        mesh.garbage_collection();  // NB this invalidates handles, so dont hold on to them
    }
}

template <typename T>
void NOpenMesh<T>::dump_border_faces(const char* msg, char side)
{
    LOG(info) << msg  ; 

    typedef NOpenMesh<T> Mesh ; 

    Mesh* a_mesh = NULL  ;
    Mesh* b_mesh = NULL  ;
    assert( side == 'L' || side == 'R' );

    switch(side)
    {
       case 'L':{  
                   a_mesh = leftmesh ; 
                   b_mesh = rightmesh ; 
                }
                break ;

       case 'R':{  
                   a_mesh = rightmesh ; 
                   b_mesh = leftmesh ; 
                }
                break ;
    }

    const nnode* a_node = a_mesh->node ; 
    const nnode* b_node = b_mesh->node ; 

    std::function<float(float,float,float)> a_sdf = a_node->sdf() ; 
    std::function<float(float,float,float)> b_sdf = b_node->sdf() ; 


    typedef typename T::FaceHandle          FH ; 
    typedef typename T::VertexHandle        VH ; 
    typedef typename T::EdgeHandle          EH ; 
    typedef typename T::HalfedgeHandle      HEH ; 
    typedef typename T::FaceIter            FI ; 
    typedef typename T::ConstFaceVertexIter FVI ; 
    typedef typename T::ConstFaceEdgeIter   FEI ; 
    typedef typename T::ConstFaceHalfedgeIter   FHI ; 
    typedef typename T::Point               P ; 

    //OpenMesh::FPropHandleT<int> f_inside_other ;
    //assert(a_mesh->mesh.get_property_handle(f_inside_other, F_INSIDE_OTHER));

    //OpenMesh::VPropHandleT<nuv> v_parametric;
    //assert(a_mesh->mesh.get_property_handle(v_parametric, V_PARAMETRIC));


    for( FI f=a_mesh->mesh.faces_begin() ; f != a_mesh->mesh.faces_end(); ++f ) 
    {
        const FH& fh = *f ;  
        int _f_inside_other = a_mesh->mesh.property(prop.f_inside_other, fh) ; 
        if(!is_border_face(_f_inside_other)) continue ; 
            
        std::cout << "facemask:" << _f_inside_other << std::endl ; 

        // a_mesh edges along which b_sdf changes sign can be bisected 
        // (can treat as unary functions as only one 
        //  parameter will vary along the parametric edge)
        //  does that stay true above the leaves ? need to arrange for it to stay true... 

      
        VH vh[2] ;  
        nuv uv[2] ; 
        glm::vec3 a_pos[2] ;
        P pt[2];
        float _a_sdf[2] ; 
        float _b_sdf[2] ; 
        bool pmatch[2] ; 


        for(FHI fhe=a_mesh->mesh.cfh_iter(fh) ; fhe.is_valid() ; fhe++) 
        {
            const HEH& heh = *fhe ; 
            vh[0] = a_mesh->mesh.from_vertex_handle( heh );
            vh[1] = a_mesh->mesh.to_vertex_handle( heh );

            uv[0] = a_mesh->mesh.property(prop.v_parametric, vh[0]) ; 
            uv[1] = a_mesh->mesh.property(prop.v_parametric, vh[1]) ; 

            a_pos[0] = a_node->par_pos( uv[0] );
            a_pos[1] = a_node->par_pos( uv[1] );

            _a_sdf[0] = a_sdf( a_pos[0].x, a_pos[0].y, a_pos[0].z );
            _a_sdf[1] = a_sdf( a_pos[1].x, a_pos[1].y, a_pos[1].z );

            _b_sdf[0] = b_sdf( a_pos[0].x, a_pos[0].y, a_pos[0].z );
            _b_sdf[1] = b_sdf( a_pos[1].x, a_pos[1].y, a_pos[1].z );

            pt[0] = a_mesh->mesh.point(vh[0]);
            pt[1] = a_mesh->mesh.point(vh[1]);

            pmatch[0] = pt[0][0] == a_pos[0][0] && pt[0][1] == a_pos[0][1] && pt[0][2] == a_pos[0][2] ;
            pmatch[1] = pt[1][0] == a_pos[1][0] && pt[1][1] == a_pos[1][1] && pt[1][2] == a_pos[1][2] ;

            assert( _a_sdf[0] == 0.f );
            assert( _a_sdf[1] == 0.f );


            std::cout << " heh " << heh
                      << " vh " << vh[0] << " -> " << vh[1]
                      << " uv " << uv[0].desc() << " -> " << uv[1].desc() 
                      << " a_pos " << glm::to_string(a_pos[0]) << " -> " << glm::to_string(a_pos[1]) 
                      << " pmatch[0] " << pmatch[0]
                      << " pmatch[1] " << pmatch[1]
                      << " _b_sdf " 
                      << std::setw(15) << std::setprecision(3) << std::fixed << _b_sdf[0]
                      << " -> " 
                      << std::setw(15) << std::setprecision(3) << std::fixed << _b_sdf[1]
                      << std::endl ;  

        }
    }
}


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
    LOG(info) << "dump START" ; 
    LOG(info) << msg << " " << brief() ; 
    desc.dump_vertices();
    desc.dump_faces();
    LOG(info) << "dump DONE" ; 
}


template <typename T>
std::string NOpenMesh<T>::brief()
{
    return desc.desc_euler();
}



// NTriSource interface

template <typename T>
unsigned NOpenMesh<T>::get_num_tri() const
{
    return mesh.n_faces();
}
template <typename T>
unsigned NOpenMesh<T>::get_num_vert() const
{
    return mesh.n_vertices();
}

template <typename T>
void NOpenMesh<T>::get_vert( unsigned i, glm::vec3& v   ) const
{
    typedef typename T::Point          P ; 
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 

    const VH& vh = mesh.vertex_handle(i) ;
    const P& p = mesh.point(vh); 

    v.x = p[0] ; 
    v.y = p[1] ; 
    v.z = p[2] ; 
}

template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t   ) const
{
    typedef typename T::VertexHandle   VH ; 
    typedef typename T::FaceHandle     FH ; 
    typedef typename T::ConstFaceVertexIter FVI ; 

    const FH& fh = mesh.face_handle(i) ;

    assert( mesh.valence(fh) == 3 ); 

    int n = 0 ; 
    for(FVI fv=mesh.cfv_iter(fh) ; fv.is_valid() ; fv++) 
    { 
        const VH& vh = *fv ; 
        t[n++] = vh.idx() ;
    }
    assert(n == 3);

}

template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t, glm::vec3& a, glm::vec3& b, glm::vec3& c ) const
{
    get_tri(i, t );

    get_vert(t.x, a );
    get_vert(t.y, b );
    get_vert(t.z, c );
}


// debug shapes

template <typename T>
NOpenMesh<T>* NOpenMesh<T>::cube(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    m->build.add_cube(verbosity);
    m->check();
    return m ; 
}
 
template <typename T>
NOpenMesh<T>* NOpenMesh<T>::tetrahedron(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    m->build.add_tetrahedron(verbosity);
    //std::cout << m->desc.desc() << std::endl ; 
    m->check();
    return m ; 
}

template <typename T>
NOpenMesh<T>* NOpenMesh<T>::hexpatch(int level, int verbosity, int ctrl)
{
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl ); 
    m->build.add_hexpatch(verbosity);
    m->check();
    return m ; 
}

template struct NOpenMesh<NOpenMeshType> ;


