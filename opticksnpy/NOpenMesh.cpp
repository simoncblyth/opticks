//  grab lower dependency pieces from openmeshrap MWrap.cc as needed


#include <limits>
#include <iostream>
#include <sstream>


#include "PLOG.hh"
#include "NGLM.hpp"
#include "Nuv.hpp"
#include "NNode.hpp"
#include "NCSGBSP.hpp"

#include "NOpenMesh.hpp"
#include "NOpenMeshProp.hpp"
#include "NOpenMeshDesc.hpp"
#include "NOpenMeshFind.hpp"
#include "NOpenMeshBuild.hpp"


#include <OpenMesh/Tools/Utils/MeshCheckerT.hh>



template <typename T>
const char* NOpenMesh<T>::COMBINE_HYBRID_ = "COMBINE_HYBRID" ; 

template <typename T>
const char* NOpenMesh<T>::COMBINE_CSGBSP_ = "COMBINE_CSGBSP" ; 

template <typename T>
const char* NOpenMesh<T>::MeshModeString(NOpenMeshMode_t meshmode)
{
    const char* s = NULL ; 
    if(     meshmode & COMBINE_HYBRID) s = COMBINE_HYBRID_  ;
    else if(meshmode & COMBINE_CSGBSP) s = COMBINE_CSGBSP_  ;
    return s ; 
}


template <typename T>
NOpenMesh<T>* NOpenMesh<T>::spawn(const nnode* subnode)
{
    return new NOpenMesh<T>(subnode, level, verbosity, ctrl, cfg.cfg, meshmode) ;
}

template <typename T>
NOpenMesh<T>::NOpenMesh(const nnode* node, int level, int verbosity, int ctrl, const char* polycfg, NOpenMeshMode_t meshmode )
    :
    cfg(polycfg),
    prop(mesh),
    desc(mesh, prop),
    find(mesh, cfg, prop, verbosity, node),
    build(mesh, cfg, prop, desc, find, verbosity),
    subdiv(mesh, cfg, prop, desc, find, build, verbosity),

    node(node), 
    level(level), 
    verbosity(verbosity), 
    ctrl(ctrl), 
    meshmode(meshmode),
    leftmesh(NULL),
    rightmesh(NULL),
    lfrontier(NULL),
    rfrontier(NULL)
{
    unsigned omv = NOpenMeshEnum::OpenMeshVersion();
    std::cout << "OpenMeshVersion " << std::hex << omv << std::dec << std::endl ; 

    init();
}


template <typename T>
void NOpenMesh<T>::init()
{
    if(!node) return ; 

    if(verbosity > 4)
    LOG(info) << "NOpenMesh<T>::init()"
              << " node.label " << ( node->label ? node->label : "-" )
              << " meshmode " << meshmode 
              << " MeshModeString " << MeshModeString(meshmode) 
              << cfg.brief()
              ; 

    build_csg();
    check();
}


template <typename T>
void NOpenMesh<T>::check()
{
    assert(OpenMesh::Utils::MeshCheckerT<T>(mesh).check()) ;
    if(verbosity > 3)
    LOG(info) << "NOpenMesh<T>::check OK" ; 
}

template <typename T>
void NOpenMesh<T>::subdiv_test()
{
    unsigned nloop0 = find.find_boundary_loops() ;
    if(verbosity > 0)
    LOG(info) << "subdiv_test START " 
              << " ctrl " << ctrl 
              << " cfg " << cfg.desc()
              << " verbosity " << verbosity
              << " nloop0 " << nloop0
              << desc.desc_euler()
              ;
    if(verbosity > 4) std::cout << desc.faces() << std::endl ;  


    for(int i=0 ; i < cfg.numsubdiv ; i++)
    {
        subdiv.sqrt3_refine( FIND_ALL_FACE, -1 );
    }

    unsigned nloop1 = find.find_boundary_loops() ;
    if(verbosity > 0)
    LOG(info) << "subdiv_test DONE " 
              << " ctrl " << ctrl 
              << " verbosity " << verbosity
              << " nloop1 " << nloop1
              << desc.desc_euler()
              ;

    if(verbosity > 4) std::cout << desc.faces() << std::endl ;  
}


template <typename T>
void NOpenMesh<T>::build_csg()
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
        build.add_parametric_primitive(node, level, ctrl ); // adds unique vertices and faces to build out the parametric mesh  
        build.euler_check(node, level);
    }
    else
    {
        assert(node->type == CSG_UNION || node->type == CSG_INTERSECTION || node->type == CSG_DIFFERENCE );
        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << cfg.brief()
                  << " making leftmesh rightmesh "
                  << " meshmode " << meshmode
                  << " COMBINE_HYBRID " << COMBINE_HYBRID
                  << " COMBINE_CSGBSP " << COMBINE_CSGBSP
                  << " MeshModeString " << MeshModeString(meshmode)
                  ; 

        leftmesh = spawn(node->left) ; 
        rightmesh = spawn(node->right) ; 


        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << " START "
                  ;

        std::cout << " leftmesh  " << leftmesh->brief() << std::endl ; 
        std::cout << " rightmesh " << rightmesh->brief() << std::endl ; 


        if( meshmode & COMBINE_HYBRID )
        {
            combine_hybrid();
        }
        else if( meshmode & COMBINE_CSGBSP )
        {
            combine_csgbsp() ; 
        }
        else
        {
            assert(0) ; 
        }

        LOG(info) 
                  << "NOpenMesh<T>::build_csg" 
                  << " DONE "
                  ;

        std::cout << " leftmesh  " << leftmesh->brief() << std::endl ; 
        std::cout << " rightmesh " << rightmesh->brief() << std::endl ; 
        std::cout << " combined " << brief() << std::endl ; 
    }
}



template <typename T>
void NOpenMesh<T>::combine_csgbsp()
{
    NCSGBSP csgbsp( leftmesh, rightmesh, node->type );
    build.copy_faces( &csgbsp );
}


template <typename T>
void NOpenMesh<T>::combine_hybrid( )
{
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

    LOG(info) << "combine_hybrid" 
              << " leftmesh " << leftmesh
              << " rightmesh " << rightmesh
               ;

    // initial meshes should be closed with no boundary loops
    assert( leftmesh->find.find_boundary_loops() == 0) ;   
    assert( rightmesh->find.find_boundary_loops() == 0) ;  

    leftmesh->build.mark_faces( node->right );
    rightmesh->build.mark_faces( node->left );


    /*
      hmm refinement expects to find a sidecorner to seed traverse... but 
      leftmesh and rightmesh are complete meshes...

      * Where to refine ? whilst still in the closed initial parametric meshes, 
        as separate frontier patches or once in combined ?

      * In general case there can be multiple frontier patches
        (ie contiguous patches of tris with verts that are not pure left/right) 
        from left and right, that need to be paired together and zippered 
        and the combined into composite ... 

        Each frontier patch "ring" should have two boundary loops, SDF 
        value can distinguish.
        
        1. towards the retained parametric mesh which needs to be unchanged
        2. towards intersection that needs zippering


        Trying refinement of frontiers shows subdiv issues, non-manifold meshes...
        TODO:
           look into the contiguous order being used... 
 
    */

    if(cfg.numsubdiv > 0)
    {
        NOpenMeshFindType sel = FIND_FACEMASK_FACE ;

        leftmesh->subdiv.sqrt3_refine( sel , -1 );  // refining frontier tris
        rightmesh->subdiv.sqrt3_refine( sel , -1 );

        leftmesh->build.mark_faces( node->right );
        rightmesh->build.mark_faces( node->left );

        // despite subdiv should still be closed zero boundary meshes
        assert( leftmesh->find.find_boundary_loops() == 0) ;   
        assert( rightmesh->find.find_boundary_loops() == 0) ;  
    }



    // copying subsets of leftmesh and rightmesh faces into this one 
    // selected via the NOpenMeshPropType

    if(node->type == CSG_UNION)
    {
        //NOpenMeshPropType prop = PROP_OUTSIDE_OTHER ;
        NOpenMeshPropType prop = PROP_FRONTIER ;

        build.copy_faces( leftmesh,  prop );
        //build.copy_faces( rightmesh, prop );

        //subdiv.sqrt3_refine( FIND_ALL_FACE , -1 );  // test refining copied over frontier tris

    }
    else if(node->type == CSG_INTERSECTION)
    {
        build.copy_faces( leftmesh,  PROP_INSIDE_OTHER );  
        build.copy_faces( rightmesh, PROP_INSIDE_OTHER );
    }
    else if(node->type == CSG_DIFFERENCE )
    {
        build.copy_faces( leftmesh,  PROP_OUTSIDE_OTHER );
        build.copy_faces( rightmesh, PROP_INSIDE_OTHER  );
    }



    int nloop = find.find_boundary_loops() ;
    // hmm expecting 2, but thats geometry specific

    find.dump_boundary_loops("find.dump_boundary_loops", true );


    if(verbosity > 0)
    LOG(info) << "combine_hybrid"
              << " boundary_loops " << nloop 
               ;    


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


    for( FI f=a_mesh->mesh.faces_begin() ; f != a_mesh->mesh.faces_end(); ++f ) 
    {
        const FH fh = *f ;  
        if(!prop.is_facemask_face(fh, -1)) continue ; 
            
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

            uv[0] = a_mesh->prop.get_uv(vh[0]) ; 
            uv[1] = a_mesh->prop.get_uv(vh[1]) ; 

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
    const VH vh = mesh.vertex_handle(i) ;
    const P  p = mesh.point(vh); 

    v.x = p[0] ; 
    v.y = p[1] ; 
    v.z = p[2] ; 
}

template <typename T>
void NOpenMesh<T>::get_normal( unsigned i, glm::vec3& n   ) const
{
    const VH vh = mesh.vertex_handle(i) ;
    const P  nrm = mesh.normal(vh); 

    n.x = nrm[0] ; 
    n.y = nrm[1] ; 
    n.z = nrm[2] ; 
}



template <typename T>
void NOpenMesh<T>::get_uv( unsigned i, glm::vec3& ouv   ) const
{
    const VH vh = mesh.vertex_handle(i) ;
    nuv uv = prop.get_uv( vh );

    ouv.x = uv.u() ;   // <-- currently the integer values
    ouv.y = uv.v() ; 
    ouv.z = uv.s() ; 
}


template <typename T>
void NOpenMesh<T>::get_tri( unsigned i, glm::uvec3& t   ) const
{
    //typedef typename T::VertexHandle   VH ; 
    //typedef typename T::FaceHandle     FH ; 
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
NOpenMesh<T>* NOpenMesh<T>::BuildTest(int level, int verbosity, int ctrl, const char* polycfg)
{
    NOpenMeshMode_t meshmode = COMBINE_HYBRID ;
    NOpenMesh<T>* m = new NOpenMesh<T>(NULL, level, verbosity, ctrl, polycfg, meshmode ); 
    switch(ctrl)
    {
       case   3: m->build.add_tripatch()       ; break ; 
       case   4: m->build.add_tetrahedron()    ; break ; 
       case   6: m->build.add_cube()           ; break ; 
       case  66: m->build.add_hexpatch(true)   ; break ; 
       case 666: m->build.add_hexpatch(false)  ; break ; 
    }
    m->check();
    return m ; 
}


template struct NOpenMesh<NOpenMeshType> ;


