#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>


// #include "G4VisExtent.hh"  

#include "G4VSolid.hh"
#include "G4Polyhedron.hh"

#include "X4SolidExtent.hh"
#include "X4Mesh.hh"

#include "YOGMaker.hh"
#include "GMesh.hh"
#include "GMeshMaker.hh"
#include "BFile.hh"
#include "BStr.hh"
#include "SDirect.hh"

#include "NBBox.hpp"
#include "NTrianglesNPY.hpp"
#include "NPY.hpp"

#include "PLOG.hh"


/**
X4Mesh::Placeholder
--------------------

Converts G4VSolid into placeholder bounding box GMesh, the
solid name is prefixed with "PLACEHOLDER_" to make this clear. 

**/

GMesh* X4Mesh::Placeholder(const G4VSolid* solid ) //static
{

/*
    G4VisExtent ve = solid->GetExtent();
    //LOG(info) << " visExtent " << ve ; 
 
    nbbox bb = make_bbox( 
                   ve.GetXmin(), ve.GetYmin(), ve.GetZmin(), 
                   ve.GetXmax(), ve.GetYmax(), ve.GetZmax(),  false );  
*/

    nbbox* bb = X4SolidExtent::Extent(solid) ; 

    NTrianglesNPY* tris = NTrianglesNPY::box(*bb) ;     
    GMesh* mesh = GMeshMaker::make_mesh(tris->getTris());

    NVtxIdx vtxidx ;
    tris->to_vtxidx(vtxidx);

    vtxidx.idx->reshape(-1,1); // YOGMaker expects this shape for indices

    mesh->m_x4src_vtx = vtxidx.vtx ; 
    mesh->m_x4src_idx = vtxidx.idx ; 
    mesh->m_g4vsolid = (void*)solid ; 

    const std::string& soName = solid->GetName(); 
    const char* name = BStr::concat("PLACEHOLDER_", soName.c_str(), NULL );  
    mesh->setName(name); 

    return mesh ; 
}


GMesh* X4Mesh::Convert(const G4VSolid* solid ) //static
{
    X4Mesh xm(solid); 
    GMesh* mesh = xm.getMesh();
    mesh->m_g4vsolid = (void*)solid ; 
    return mesh ; 
}

X4Mesh::X4Mesh( const G4VSolid* solid ) 
   :
   m_solid(solid),
   m_polyhedron(NULL),
   m_vtx(NULL),
   m_raw(NULL),
   m_tri(NULL),
   m_mesh(NULL),
   m_verbosity(0)
{
   init() ;
}

GMesh* X4Mesh::getMesh() const 
{
   return m_mesh ; 
}

void X4Mesh::init()
{
    polygonize();
    collect();
    makemesh();

    assert(m_mesh);
    const std::string& soName = m_solid->GetName(); 
    m_mesh->setName(soName.c_str()); 
}




std::string X4Mesh::desc() const 
{
    std::stringstream ss ; 
    ss << "X4Mesh"
       << " vtx " << ( m_vtx ? m_vtx->getShapeString() : "-" )
       << " raw " << ( m_raw ? m_raw->getShapeString() : "-" )
       << " tri " << ( m_tri ? m_tri->getShapeString() : "-" )
       ;
    return ss.str();
}

void X4Mesh::polygonize()
{
    G4bool create = true ; 
    G4int noofsides = 24 ;
    G4Polyhedron::SetNumberOfRotationSteps(noofsides);

    std::stringstream coutbuf;
    std::stringstream cerrbuf;
    {   
       cout_redirect out(coutbuf.rdbuf());
       cerr_redirect err(cerrbuf.rdbuf());
       if( create ){         
           m_polyhedron = m_solid->CreatePolyhedron ();  // always create a new poly   
       } else {
           m_polyhedron = m_solid->GetPolyhedron ();     // if poly created already and no parameter change just provide that one 
       }   
    }   

    std::string cout_ = coutbuf.str() ; 
    std::string cerr_ = cerrbuf.str() ; 

    if(cout_.size() > 0) LOG(info) << cout_ ; 
    if(cerr_.size() > 0) LOG(warning) << cerr_ ; 

    std::string polysmry ; 
    { 
       std::stringstream ss ;
       ss << "v " << m_polyhedron->GetNoVertices() << " " ;
       ss << "f " << m_polyhedron->GetNoFacets() << " " ;
       ss << "cout " << cout_.size() << " " ; 
       ss << "cerr " << cerr_.size() << " " ; 
       polysmry = ss.str();
    }
   
    LOG(debug) << polysmry ; 
}


void X4Mesh::collect()
{
    G4int nv = m_polyhedron->GetNoVertices();
    G4int nf = m_polyhedron->GetNoFacets();

    // suspect that nv is 1 bigger than it should be,
    // and the top is vertex is never filled 

    m_vtx = NPY<float>::make(nv, 3) ; 
    m_vtx->zero();

    m_raw = NPY<unsigned>::make(nf, 4) ;  // 4 slots available to handle quads
    m_raw->zero();

    for (int i = 0 ; i < nv ; i++) collect_vtx(i+1) ; 
    for (int i = 0 ; i < nf ; i++) collect_raw(i+1) ; 

    collect_tri(); 

}

void X4Mesh::collect_vtx(int ivert)
{
    G4Point3D vtx = m_polyhedron->GetVertex(ivert); // ivert is 1-based index from 1 to nv

    m_vtx->setValue( ivert-1, 0, 0, 0, vtx.x() );    
    m_vtx->setValue( ivert-1, 0, 0, 1, vtx.y() );    
    m_vtx->setValue( ivert-1, 0, 0, 2, vtx.z() );    
    //                        ^^^^ are these zeros a hangover from former different shape of m_vtx 
}

void X4Mesh::collect_raw(int iface)
{
    G4int nedge;
    G4int ivertex[4];
    G4int iedgeflag[4];
    G4int ifacet[4];

    m_polyhedron->GetFacet(iface, nedge, ivertex, iedgeflag, ifacet);

    if(m_verbosity > 1)
    {
        std::cout 
            << " iface - 1 " << iface - 1 
            << " nedge " << nedge 
            << " verts ( " 
            ;
    }

    assert( nedge == 3 || nedge == 4 ); 

    if(m_verbosity > 1 )
    {
        G4int iedge = 0;
        for(iedge = 0; iedge < nedge; ++iedge) {
            std::cout << ivertex[iedge] - 1 << " " ;   
        }   
        std::cout << " ) " << std::endl ; 
    }


    // collect raw indices, allowing tris or quads 
    // vertex indices are 1-based, for tris the 4th slot is zero 

    m_raw->setQuad( iface-1, 0, 0, 
            ivertex[0] ,
            ivertex[1] , 
            ivertex[2] ,
            nedge == 4 ? ivertex[3]  : 0 ) ;
}


void X4Mesh::collect_tri()
{
    unsigned nf = m_raw->getShape(0);  

    unsigned ntri = 0 ; 
    for(unsigned i=0 ; i < nf ; i++)
    {
         glm::uvec4 raw = m_raw->getQuadU(i, 0) ; 
         ntri += ( raw.w == 0 ? 1 : 2 ) ;     // 2 tri for a quad
    }

    m_tri = NPY<unsigned>::make( ntri, 3 ) ;  
    m_tri->zero();

    unsigned jtri = 0 ; 
    for(unsigned i=0 ; i < nf ; i++)
    {
         glm::uvec4 raw = m_raw->getQuadU(i, 0) ; 

/*
         std::cout 
             << std::setw(6) << raw.x   
             << std::setw(6) << raw.y
             << std::setw(6) << raw.z   
             << std::setw(6) << raw.w
             << std::endl ;    
*/
         if(raw.w == 0)
         {
             m_tri->setValue( jtri, 0, 0,  0, raw.x-1 ) ;
             m_tri->setValue( jtri, 0, 0,  1, raw.y-1 ) ;
             m_tri->setValue( jtri, 0, 0,  2, raw.z-1 ) ;
             jtri += 1 ;  
         } 
         else if ( raw.w != 0 )
         {
             m_tri->setValue( jtri, 0, 0,  0, raw.x-1 ) ;
             m_tri->setValue( jtri, 0, 0,  1, raw.y-1 ) ;
             m_tri->setValue( jtri, 0, 0,  2, raw.z-1 ) ;
             jtri += 1 ;  

             m_tri->setValue( jtri, 0, 0,  0, raw.x-1 ) ;
             m_tri->setValue( jtri, 0, 0,  1, raw.z-1 ) ;
             m_tri->setValue( jtri, 0, 0,  2, raw.w-1 ) ;
             jtri += 1 ;  
         }
    }

    assert( jtri == ntri );

    /*

           w-------z
           |       |
           |       |
           |       |
           x-------y

           +-------z
           |     / |
           |   /   |
           | /     |
           x-------y

           w-------z
           |     / |
           |   /   |
           | /     |
           x-------+

    */
}


void X4Mesh::save(const char* path) const 
{
    LOG(info) << " saving to " << path ;     

    std::string dir_ = BFile::ParentDir(path); 
    const char* dir = dir_.c_str();

    m_vtx->save(dir,"vtx.npy"); 
    m_raw->save(dir,"raw.npy"); 
    m_tri->save(dir,"tri.npy"); 

    m_mesh->save(dir); 

    GMesh* mesh = getMesh();
    const NPY<float>* vtx = mesh->m_x4src_vtx ; 
    const NPY<unsigned>* idx = mesh->m_x4src_idx ; 

    YOG::Maker::SaveToGLTF( vtx, idx, path ); 
}


void X4Mesh::makemesh()
{
    bool via_tris = false ; 
    if( via_tris ) 
    {
        NTrianglesNPY* tris = NTrianglesNPY::from_indexed(m_vtx, m_tri);
        m_mesh = GMeshMaker::make_mesh(tris->getTris());
    }
    else
    {
        m_mesh = GMeshMaker::make_mesh(m_vtx, m_tri, 0 );
    }
}




