#include "CFG4_BODY.hh"
#include "CMaker.hh"

// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NParameters.hpp"
#include "NPrimitives.hpp"
#include "GLMFormat.hpp"

// ggeo-
#include "GCSG.hh"

// g4-
#include "G4Sphere.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Trd.hh"
#include "G4Torus.hh"

#include "G4TriangularFacet.hh"
#include "G4QuadrangularFacet.hh"
#include "G4TessellatedSolid.hh"

#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"

#include "G4UnionSolid.hh"
#include "G4SubtractionSolid.hh"
#include "G4IntersectionSolid.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "PLOG.hh"



CMaker::CMaker(Opticks* ok, int verbosity) 
   :
   m_ok(ok),
   m_verbosity(verbosity)
{
}   

std::string CMaker::LVName(const char* shapename, int idx)
{
    std::stringstream ss ; 
    ss << shapename << "_log" ; 
    if(idx > -1) ss << idx ; 
    return ss.str();
}

std::string CMaker::PVName(const char* shapename, int idx)
{
    std::stringstream ss ; 
    ss << shapename << "_phys" ; 
    if(idx > -1) ss << idx ; 
    return ss.str();
}


G4VSolid* CMaker::makeSolid(const NCSG* csg)
{
    nnode* root_ = csg->getRoot();

    G4VSolid* root = makeSolid_r(root_);

    return root  ; 
}

G4VSolid* CMaker::makeSolid_r(const nnode* node)
{
    // hmm rmin/rmax is handled as a CSG subtraction
    // so could collapse some operators into primitives

    G4VSolid* result = NULL ; 

    const char* name = node->csgname();

    if( node->is_primitive() )
    {
        result = ConvertPrimitive(node);        
    }
    else if(node->is_operator())
    {
        G4VSolid* left = makeSolid_r(node->left);
        G4VSolid* right = makeSolid_r(node->right);

        bool left_transform = node->left->gtransform ? !node->left->gtransform->is_identity() : false ;  
        if(left_transform)
        {
            LOG(info) << " unexpected non-identity left transform "
                      << gformat(node->left->gtransform->t )
                      ;
        }
        assert(left_transform == false);

        G4Transform3D* rtransform = ConvertTransform(node->right->gtransform->t);

        if(node->type == CSG_UNION)
        {
            G4UnionSolid* uso = new G4UnionSolid( name, left, right, *rtransform );
            result = uso ; 
        }
        else if(node->type == CSG_INTERSECTION)
        {
            G4IntersectionSolid* iso = new G4IntersectionSolid( name, left, right, *rtransform );
            result = iso ; 
        }
        else if(node->type == CSG_DIFFERENCE)
        {
            G4SubtractionSolid* sso = new G4SubtractionSolid( name, left, right, *rtransform );
            result = sso ; 
        }
    }
    return result  ; 
}



class ConvertedG4Transform3D : public G4Transform3D 
{
   // inherit in order to use the protected ctor
    public:
        ConvertedG4Transform3D(const glm::mat4& m) :
              G4Transform3D( 
                               m[0].x, m[0].y, m[0].z, m[0].w ,
                               m[1].x, m[1].y, m[1].z, m[1].w ,
                               m[2].x, m[2].y, m[2].z, m[2].w ) 
        {}
};
                


G4Transform3D* CMaker::ConvertTransform(const glm::mat4& t) // static
{
    glm::mat4 tt = nglmext::make_transpose(t);
    ConvertedG4Transform3D* ctr = new ConvertedG4Transform3D(tt);
    return ctr  ; 
}




/**

g4-;g4-cls G4TessellatedSolid
g4-;g4-cls G4TriangularFacet
g4-;g4-cls G4VFacet

**/
 
G4VSolid* CMaker::ConvertConvexPolyhedron(const nnode* node) // static
{
    NParameters* meta = node->meta ;  
    assert(meta);
    std::string src_type = meta->getStringValue("src_type");
   
    // see src_type args of the ConvexPolyhedronSrc in opticks/analytic/prism.py 
    bool prism = src_type.compare("prism")==0 ;
    bool segment = src_type.compare("segment")==0 ;
    bool icosahedron =  src_type.compare("icosahedron") == 0 ; 
    bool cubeplanes =  src_type.compare("cubeplanes") == 0 ; 
 
    bool supported = prism || segment || icosahedron || cubeplanes ;
    if(!supported) 
         LOG(fatal) << " src_type not supprted " <<  src_type ;

    assert( supported  );
  

    G4VSolid* result = NULL ; 
    const char* name = node->csgname();
    assert(name);

    nconvexpolyhedron* n = (nconvexpolyhedron*)node ; 

    const std::vector<glm::ivec4>& faces = n->srcfaces ;  
    const std::vector<glm::vec3>&  verts = n->srcverts ;  

    unsigned nv = verts.size();
    unsigned nf = faces.size();

    LOG(info) 
         << " faces " << nf
         << " verts " << nv
         ;

    std::vector<G4ThreeVector> v ; 
    for(unsigned i=0 ; i < nv ; i++) v.push_back( G4ThreeVector(verts[i].x, verts[i].y, verts[i].z ) );

    G4TessellatedSolid* te = new G4TessellatedSolid(name);
    for(unsigned i=0 ; i < nf ; i++)
    {
        glm::ivec4 face = faces[i] ; 
        if( face.w == -1 )
        {
            assert( unsigned(face.x) < nv );
            assert( unsigned(face.y) < nv );
            assert( unsigned(face.z) < nv );

            G4TriangularFacet* tf = new G4TriangularFacet( v[face.x], v[face.y], v[face.z], ABSOLUTE ) ; 
            te->AddFacet(tf);
        }
        else
        {
            assert( unsigned(face.x) < nv );
            assert( unsigned(face.y) < nv );
            assert( unsigned(face.z) < nv );
            assert( unsigned(face.w) < nv );

            G4QuadrangularFacet* qf = new G4QuadrangularFacet( v[face.x], v[face.y], v[face.z], v[face.w],  ABSOLUTE ) ; 
            te->AddFacet(qf);
        }
    }
    te->SetSolidClosed(true) ;
    result = te ;   
    return result ; 
}


G4VSolid* CMaker::ConvertPrimitive(const nnode* node) // static
{
    /*
    G4 has inner imps that would allow some Opticks operators to be
    expressed as G4 primitives. 
    */

    G4VSolid* result = NULL ; 
    const char* name = node->csgname();
    assert(name);


    // cf NCSG::import_primitive
    if(node->type == CSG_SPHERE )
    {
        nsphere* n = (nsphere*)node ; 
        G4Sphere* sp = new G4Sphere( name, 0., n->radius(), 0., twopi, 0., pi);  
        result = sp ; 
    }
    else if(node->type == CSG_ZSPHERE)
    {
        nzsphere* n = (nzsphere*)node ; 

        double innerRadius = 0. ;
        double outerRadius = n->radius() ;
        double startPhi = 0. ; 
        double deltaPhi = twopi ; 
        double startTheta = n->startTheta() ; 
        double deltaTheta = n->deltaTheta() ; 
        double endTheta = startTheta + deltaTheta ; 

        LOG(error) << "CSG_ZSPHERE"
                   << " pi " << pi
                   << " innerRadius " << innerRadius
                   << " outerRadius " << outerRadius
                   << " startTheta " << startTheta
                   << " deltaTheta " << deltaTheta
                   << " endTheta " << endTheta
                   ;
        assert( startTheta <= pi && startTheta >= 0.);
        assert( deltaTheta <= pi && deltaTheta >= 0.);
        assert( endTheta <= pi && endTheta >= 0.);

        G4Sphere* sp = new G4Sphere( name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);  
        result = sp ; 
    }
    else if(node->type == CSG_BOX || node->type == CSG_BOX3)
    {
        nbox* n = (nbox*)node ; 
        glm::vec3 halfside = n->halfside();

        G4Box* bx = new G4Box( name, halfside.x, halfside.y, halfside.z );
        result = bx ; 
    }
    else if(node->type == CSG_TORUS)
    {
        ntorus* n = (ntorus*)node ; 

        double innerRadius = 0. ;
        double outerRadius = n->rminor();
        double sweptRadius = n->rmajor(); 
        double startPhi = 0. ;
        double deltaPhi = twopi ;

        G4Torus* ts = new G4Torus( name, innerRadius, outerRadius, sweptRadius, startPhi, deltaPhi );
        result = ts ;  
    }
    else if(node->type == CSG_CYLINDER)
    {
        ncylinder* n = (ncylinder*)node ; 

        float z1 = n->z1() ; 
        float z2 = n->z2() ;
        assert( z2 > z1 && z2 == -z1 ); 
        float hz = fabs(z1) ;

        double innerRadius = 0. ;
        double outerRadius = n->radius() ;
        double zHalfLength = hz ;  // hmm will need transforms for nudged ?
        double startPhi = 0. ; 
        double deltaPhi = twopi ; 

        G4Tubs* tb = new G4Tubs( name, innerRadius, outerRadius, zHalfLength, startPhi, deltaPhi );
        result = tb ; 
    }
    else if(node->type == CSG_DISC)
    {
        ndisc* n = (ndisc*)node ; 

        float z1 = n->z1() ; 
        float z2 = n->z2() ;
        assert( z2 > z1 && z2 == -z1 ); 
        float hz = fabs(z1) ;

        double innerRadius = 0. ;
        double outerRadius = n->radius() ;
        double zHalfLength = hz ;  // hmm will need transforms for nudged ?
        double startPhi = 0. ; 
        double deltaPhi = twopi ; 

        G4Tubs* tb = new G4Tubs( name, innerRadius, outerRadius, zHalfLength, startPhi, deltaPhi );
        result = tb ; 
    }
    else if(node->type == CSG_CONE)
    {
        ncone* n = (ncone*)node ; 

        float z1 = n->z1() ; 
        float z2 = n->z2() ;
        assert( z2 > z1 && z2 == -z1 ); 
        float hz = fabs(z1) ;

        double innerRadius1 = 0. ; 
        double innerRadius2 = 0. ; 
        double outerRadius1 = n->r1() ; 
        double outerRadius2 = n->r2() ; 
        double zHalfLength = hz  ; 
        double startPhi = 0. ; 
        double deltaPhi = twopi ; 
         
        G4Cons* cn = new G4Cons( name, innerRadius1, outerRadius1, innerRadius2, outerRadius2, zHalfLength, startPhi, deltaPhi );
        result = cn ; 
    }
    else if(node->type == CSG_TRAPEZOID || node->type == CSG_SEGMENT || node->type == CSG_CONVEXPOLYHEDRON)
    {
        NParameters* meta = node->meta ;  
        assert(meta);
        std::string src_type = meta->getStringValue("src_type");

        if(src_type.compare("trapezoid")==0)
        {
            float src_z = meta->get<float>("src_z");
            float src_x1 = meta->get<float>("src_x1");
            float src_y1 = meta->get<float>("src_y1");
            float src_x2 = meta->get<float>("src_x2");
            float src_y2 = meta->get<float>("src_y2");

            G4Trd* tr = new G4Trd( name, src_x1, src_x2, src_y1, src_y2, src_z ); 
            result = tr ; 
        }
        else
        {
            result = ConvertConvexPolyhedron( node );
        }   
    }
    else
    {
        LOG(fatal) << "CMaker::ConvertPrimitive MISSING IMP FOR  " << name ; 
        assert(0);
    }
    return result ; 
}

















