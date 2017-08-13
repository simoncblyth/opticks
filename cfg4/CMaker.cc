#include "CFG4_BODY.hh"
#include "CMaker.hh"

// npy-
#include "NGLM.hpp"
#include "NGLMExt.hpp"
#include "NCSG.hpp"
#include "NNode.hpp"
#include "NParameters.hpp"
#include "NPrimitives.hpp"

// ggeo-
#include "GCSG.hh"

// g4-
#include "G4Sphere.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Cons.hh"
#include "G4Trd.hh"

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


std::string CMaker::LVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_log" ; 
    return ss.str();
}

std::string CMaker::PVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_phys" ; 
    return ss.str();
}



G4VSolid* CMaker::makeSphere(const glm::vec4& param)
{
    G4double radius = param.w*mm ; 
    G4Sphere* solid = new G4Sphere("sphere_solid", 0., radius, 0., twopi, 0., pi);  
    return solid ; 
}

G4VSolid* CMaker::makeBox(const glm::vec4& param)
{
    G4double extent = param.w*mm ; 
    G4double x = extent;
    G4double y = extent;
    G4double z = extent;
    G4Box* solid = new G4Box("box_solid", x,y,z);
    return solid ; 
}

G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
{
   // hmm this is somewhat specialized to known structure of DYB PMT
   //  eg intersections are limited to 3 ?

    unsigned int nc = csg->getNumChildren(index); 
    unsigned int fc = csg->getFirstChildIndex(index); 
    unsigned int lc = csg->getLastChildIndex(index); 
    unsigned int tc = csg->getTypeCode(index);
    const char* tn = csg->getTypeName(index);

    if(m_verbosity>0)
    LOG(info) 
           << "CMaker::makeSolid (GCSG)  "
           << "  i " << std::setw(2) << index  
           << " nc " << std::setw(2) << nc 
           << " fc " << std::setw(2) << fc 
           << " lc " << std::setw(2) << lc 
           << " tc " << std::setw(2) << tc 
           << " tn " << tn 
           ;

   G4VSolid* solid = NULL ; 

   if(csg->isUnion(index))
   {
       assert(nc == 2);
       std::stringstream ss ; 
       ss << "union-ab" 
          << "-i-" << index
          << "-fc-" << fc 
          << "-lc-" << lc 
          ;
       std::string ab_name = ss.str();

       int a = fc ; 
       int b = lc ; 

       G4ThreeVector apos(csg->getX(a)*mm, csg->getY(a)*mm, csg->getZ(a)*mm); 
       G4ThreeVector bpos(csg->getX(b)*mm, csg->getY(b)*mm, csg->getZ(b)*mm);

       G4RotationMatrix ab_rot ; 
       G4Transform3D    ab_transform(ab_rot, bpos  );

       G4VSolid* asol = makeSolid(csg, a );
       G4VSolid* bsol = makeSolid(csg, b );

       G4UnionSolid* uso = new G4UnionSolid( ab_name.c_str(), asol, bsol, ab_transform );
       solid = uso ; 
   }
   else if(csg->isIntersection(index))
   {
       assert(nc == 3 && fc + 2 == lc );

       std::string ij_name ;      
       std::string ijk_name ;      

       {
          std::stringstream ss ; 
          ss << "intersection-ij" 
              << "-i-" << index 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ij_name = ss.str();
       }
  
       {
          std::stringstream ss ; 
          ss << "intersection-ijk" 
              << "-i-" << index 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ijk_name = ss.str();
       }


       int i = fc + 0 ; 
       int j = fc + 1 ; 
       int k = fc + 2 ; 

       G4ThreeVector ipos(csg->getX(i)*mm, csg->getY(i)*mm, csg->getZ(i)*mm); // kinda assumed 0,0,0
       G4ThreeVector jpos(csg->getX(j)*mm, csg->getY(j)*mm, csg->getZ(j)*mm);
       G4ThreeVector kpos(csg->getX(k)*mm, csg->getY(k)*mm, csg->getZ(k)*mm);

       G4VSolid* isol = makeSolid(csg, i );
       G4VSolid* jsol = makeSolid(csg, j );
       G4VSolid* ksol = makeSolid(csg, k );

       G4RotationMatrix ij_rot ; 
       G4Transform3D    ij_transform(ij_rot, jpos  );
       G4IntersectionSolid* ij_sol = new G4IntersectionSolid( ij_name.c_str(), isol, jsol, ij_transform  );

       G4RotationMatrix ijk_rot ; 
       G4Transform3D ijk_transform(ijk_rot,  kpos );
       G4IntersectionSolid* ijk_sol = new G4IntersectionSolid( ijk_name.c_str(), ij_sol, ksol, ijk_transform  );

       solid = ijk_sol ; 
   } 
   else if(csg->isSphere(index))
   {
        std::stringstream ss ; 
        ss << "sphere" 
              << "-i-" << index 
              ; 

       std::string sp_name = ss.str();

       float inner = float(csg->getInnerRadius(index)*mm) ;
       float outer = float(csg->getOuterRadius(index)*mm) ;
       float startTheta = float(csg->getStartTheta(index)*pi/180.) ;
       float deltaTheta = float(csg->getDeltaTheta(index)*pi/180.) ;

       assert(outer > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*float(pi) ; 

       LOG(info) << "CMaker::makeSolid csg Sphere"
                 << " inner " << inner 
                 << " outer " << outer
                 << " startTheta " << startTheta
                 << " deltaTheta " << deltaTheta
                 << " endTheta " << startTheta + deltaTheta
                 ;
 
       solid = new G4Sphere( sp_name.c_str(), inner > 0 ? inner : 0.f , outer, startPhi, deltaPhi, startTheta, deltaTheta  );

   }
   else if(csg->isTubs(index))
   {
        std::stringstream ss ; 
        ss << "tubs" 
              << "-i-" << index 
              ; 

       std::string tb_name = ss.str();
       float inner = 0.f ; // csg->getInnerRadius(i); kludge to avoid rejig as sizeZ occupies innerRadius spot
       float outer = float(csg->getOuterRadius(index)*mm) ;
       float sizeZ = float(csg->getSizeZ(index)*mm) ;   // half length   
       sizeZ /= 2.f ;   

       // PMT base looks too long without the halfing (as seen by photon interaction position), 
       // but tis contrary to manual http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/DetDesc/Documents/Solids.pdf

       assert(sizeZ > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*float(pi) ; 

       if(m_verbosity>0)
       LOG(info) << "CMaker::makeSolid"
                 << " name " << tb_name
                 << " inner " << inner 
                 << " outer " << outer 
                 << " sizeZ " << sizeZ 
                 << " startPhi " << startPhi
                 << " deltaPhi " << deltaPhi
                 << " mm " << mm
                 ;

       solid = new G4Tubs( tb_name.c_str(), inner > 0 ? inner : 0.f , outer, sizeZ, startPhi, deltaPhi );

   }
   else
   {
       LOG(warning) << "CMaker::makeSolid implementation missing " ; 
   }

   assert(solid) ; 
   return solid ; 
}

G4VSolid* CMaker::makeSolid(OpticksCSG_t type, const glm::vec4& param)
{
    G4VSolid* solid = NULL ; 
    switch(type)
    {
        case CSG_BOX:   solid = makeBox(param);break;
        case CSG_SPHERE:solid = makeSphere(param);break;
        case CSG_UNION:
        case CSG_INTERSECTION:
        case CSG_DIFFERENCE:
        case CSG_ZSPHERE:
        case CSG_ZLENS:
        case CSG_PMT:
        case CSG_PRISM:
        case CSG_TUBS:
        case CSG_PARTLIST:
        case CSG_CYLINDER:
        case CSG_DISC:
        case CSG_CONE:
        case CSG_MULTICONE:
        case CSG_BOX3:
        case CSG_PLANE:
        case CSG_SLAB:
        case CSG_TRAPEZOID:
        case CSG_ZERO:
        case CSG_UNDEFINED:
        case CSG_FLAGPARTLIST:
        case CSG_FLAGNODETREE:
        case CSG_FLAGINVISIBLE:
        case CSG_CONVEXPOLYHEDRON:
        case CSG_SEGMENT:
        case CSG_TORUS:
        case CSG_CUBIC:
        case CSG_ELLIPSOID:
        case CSG_HYPERBOLOID:
                         solid = NULL ; break ; 

    }
    return solid ; 
} 


G4VSolid* CMaker::makeSolid(NCSG* csg)
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

        assert(node->left->gtransform == NULL );

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

G4VSolid* CMaker::ConvertPrimitive(const nnode* node) // static
{
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
            assert(0);
        }   
    }
    else
    {
        LOG(fatal) << "CMaker::ConvertPrimitive " << name ; 
        assert(0);
    }
    return result ; 
}


