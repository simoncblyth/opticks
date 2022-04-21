#include <array>

#include "scuda.h"
#include "squad.h"
#include "stran.h"
#include "OpticksCSG.h"

#include "PLOG.hh"

#include "CSGNode.h"
#include "CSGFoundry.h"
#include "CSGMaker.h"

const plog::Severity CSGMaker::LEVEL = PLOG::EnvLevel("CSGMaker", "DEBUG" ); 

CSGMaker::CSGMaker( CSGFoundry* fd_ )
    :
    fd(fd_)
{
}

bool CSGMaker::StartsWith( const char* n, const char* q ) // static
{
    return strlen(q) >= strlen(n) && strncmp(q, n, strlen(n)) == 0 ; 
}

bool CSGMaker::CanMake(const char* qname) // static 
{
    bool found = false ; 
    std::stringstream ss(NAMES) ;    
    std::string name ; 
    while (!found && std::getline(ss, name)) if(!name.empty() && StartsWith(name.c_str(), qname)) found = true ;
    LOG(LEVEL) << " qname " << qname << " found " << found ; 
    return found ; 
}

void CSGMaker::GetNames(std::vector<std::string>& names ) // static
{
    std::stringstream ss(NAMES) ;    
    std::string name ; 
    while (std::getline(ss, name)) if(!name.empty()) names.push_back(name); 
}


const char* CSGMaker::NAMES = R"LITERAL(
JustOrb
BoxedSphere
zsph
ZSphere
cone
hype
box3
plan
Plane
slab
Slab
cyli
disc
vcub
vtet
ConvexPolyhedronCube
ConvexPolyhedronTetrahedron
elli
ubsp
ibsp
dbsp
UnionBoxSphere
UnionListBoxSphere
UnionLLBoxSphere
IntersectionBoxSphere
OverlapBoxSphere
OverlapThreeSphere
ContiguousThreeSphere
DiscontiguousThreeSphere
DiscontiguousTwoSphere
ContiguousBoxSphere
DiscontiguousBoxSphere
DifferenceBoxSphere
ListTwoBoxTwoSphere
rcyl
dcyl
icyl
iphi
ithe
ithl
bssc
)LITERAL"; 




// see CSGNode::MakeDemo for CSGNode level equivalent
CSGSolid* CSGMaker::make(const char* name)
{
    CSGSolid* so = nullptr ; 
    if(     StartsWith("JustOrb", name))     so = makeSphere(name) ;
    else if(StartsWith("BoxedSphere", name)) so = makeBoxedSphere(name) ;
    else if(StartsWith("zsph", name))     so = makeZSphere(name) ;
    else if(StartsWith("ZSphere", name))  so = makeZSphere(name) ;
    else if(StartsWith("cone", name))     so = makeCone(name) ;
    else if(StartsWith("hype", name))     so = makeHyperboloid(name) ;
    else if(StartsWith("box3", name))     so = makeBox3(name) ;
    else if(StartsWith("plan", name))     so = makePlane(name) ;
    else if(StartsWith("Plane", name))    so = makePlane(name) ;
    else if(StartsWith("slab", name))     so = makeSlab(name) ;
    else if(StartsWith("Slab", name))     so = makeSlab(name) ;
    else if(StartsWith("cyli", name))     so = makeCylinder(name) ;
    else if(StartsWith("disc", name))     so = makeDisc(name) ;
    else if(StartsWith("vcub", name))     so = makeConvexPolyhedronCube(name) ;
    else if(StartsWith("vtet", name))     so = makeConvexPolyhedronTetrahedron(name) ;
    else if(StartsWith("ConvexPolyhedronCube", name))        so = makeConvexPolyhedronCube(name) ;
    else if(StartsWith("ConvexPolyhedronTetrahedron", name)) so = makeConvexPolyhedronTetrahedron(name) ;
    else if(StartsWith("elli", name)) so = makeEllipsoid(name) ;
    else if(StartsWith("UnionBoxSphere", name))        so = makeUnionBoxSphere(name) ;
    else if(StartsWith("UnionListBoxSphere", name))    so = makeUnionListBoxSphere(name) ;
    else if(StartsWith("UnionLLBoxSphere", name))      so = makeUnionLLBoxSphere(name) ;
    else if(StartsWith("IntersectionBoxSphere", name)) so = makeIntersectionBoxSphere(name) ;
    else if(StartsWith("OverlapBoxSphere", name))      so = makeOverlapBoxSphere(name) ;
    else if(StartsWith("OverlapThreeSphere", name))    so = makeOverlapThreeSphere(name) ;
    else if(StartsWith("ContiguousThreeSphere", name))    so = makeContiguousThreeSphere(name) ;
    else if(StartsWith("DiscontiguousThreeSphere", name))    so = makeDiscontiguousThreeSphere(name) ;
    else if(StartsWith("DiscontiguousTwoSphere", name))    so = makeDiscontiguousTwoSphere(name) ;
    else if(StartsWith("ContiguousBoxSphere", name))   so = makeContiguousBoxSphere(name) ;
    else if(StartsWith("DiscontiguousBoxSphere", name))   so = makeDiscontiguousBoxSphere(name) ;
    else if(StartsWith("DifferenceBoxSphere", name))   so = makeDifferenceBoxSphere(name) ;
    else if(StartsWith("ListTwoBoxTwoSphere", name))   so = makeListTwoBoxTwoSphere(name); 
    else if(StartsWith("rcyl", name)) so = makeRotatedCylinder(name) ;
    else if(StartsWith("dcyl", name)) so = makeDifferenceCylinder(name) ;
    else if(StartsWith("icyl", name)) so = makeInfCylinder(name) ;
    else if(StartsWith("iphi", name)) so = makeInfPhiCut(name) ;
    else if(StartsWith("ithe", name)) so = makeInfThetaCut(name) ;
    else if(StartsWith("ithl", name)) so = makeInfThetaCutL(name) ;
    else if(StartsWith("bssc", name)) so = makeBoxSubSubCylinder(name) ;
    else 
    {
        LOG(fatal) << "invalid name [" << name << "]" << " expecting one of the below " << std::endl << NAMES  ; 
        LOG(error) << "perhaps you intended to convert from a G4VSolid : if so see ~/opticks/GeoChain/translate.sh " ; 
    }
    assert( so ); 
    return so ;  
}



void CSGMaker::makeDemoSolids()
{
    makeSphere(); 
    makeZSphere(); 
    makeCone(); 
    makeHyperboloid(); 
    makeBox3(); 
    makePlane(); 
    makeSlab(); 
    makeCylinder() ; 
    makeDisc(); 
    makeConvexPolyhedronCube(); 
    makeConvexPolyhedronTetrahedron(); 
    makeEllipsoid(); 
    makeUnionBoxSphere();
    makeIntersectionBoxSphere();
    makeDifferenceBoxSphere();
    makeRotatedCylinder();
    makeDifferenceCylinder();
    makeBoxSubSubCylinder();
}

void CSGMaker::makeDemoGrid()
{
    makeDemoSolids(); 
    unsigned num_solids = fd->getNumSolid(); 
    LOG(info) << " num_solids " << num_solids ; 

    float gridscale = 100.f ; 
    std::array<int,9> grid = {{ -10,11,2,  -10,11,2, -10,11,2  }} ;

    unsigned ias_idx = 0 ; 
    unsigned count = 0 ; 

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] ){
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] ){
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] ){

        qat4 instance  ;   
        instance.q3.f.x = float(i)*gridscale ; 
        instance.q3.f.y = float(j)*gridscale ; 
        instance.q3.f.z = float(k)*gridscale ; 
        
        unsigned ins_idx = fd->inst.size() ;    
        unsigned gas_idx = count % num_solids ; 

        instance.setIdentity( ins_idx, gas_idx, ias_idx );  
        fd->inst.push_back( instance );  
     
        count++ ; 
    }   
    }   
    }   
}




/**
CSGMaker::makeLayered
----------------------------

NB Each layer is a separate CSGPrim with a single CSGNode 

NB the ordering of addition is prescribed, must stick 
ridgidly to the below order of addition:

1. addSolid
2. addPrim
3. addNode

And the numbers of Prim and Node added 
must correspond to the declarations. 

Note that the CSGNode and CSGPrim can be created anytime, the 
restriction is on the order of addition to the CSGFoundry 
due to the capturing of offsets at collection. 

**/

CSGSolid* CSGMaker::makeLayered(const char* label, float outer_radius, unsigned layers )
{
    std::vector<float> radii ;
    for(unsigned i=0 ; i < layers ; i++) radii.push_back(outer_radius*float(layers-i)/float(layers)) ; 

    unsigned numPrim = layers ; 
    CSGSolid* so = fd->addSolid(numPrim, label); 
    so->center_extent = make_float4( 0.f, 0.f, 0.f, outer_radius ) ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ; 
        CSGPrim* pr = fd->addPrim(numNode, nodeOffset_ );
 
        float radius = radii[i]; 

        CSGNode* nd = nullptr ; 

        if(strcmp(label, "sphere") == 0)
        {
            nd = fd->addNode(CSGNode::Sphere(radius)); 
        }
        else if(strcmp(label, "zsphere") == 0)
        {
            nd = fd->addNode(CSGNode::ZSphere(radius, -radius/2.f , radius/2.f )); 
        }
        else
        {
            assert( 0 && "layered only implemented for sphere and zsphere currently" ); 
        } 

        // pr->setSbtIndexOffset(i) ;  // NOW done in addPrim
        pr->setAABB( nd->AABB() ); 
    }
    return so ; 
}


/**
CSGMaker::makeBoxedSphere
--------------------------

Add CSGSolid to CSGFoundry that is comprised of two CSGPrim, 
each with a single CSGNode.  

**/

CSGSolid* CSGMaker::makeBoxedSphere(const char* label)
{
    float halfside = 100.f ; 
    float radius   = halfside/2.f ; 

    unsigned numPrim = 2 ; 
    CSGSolid* so = fd->addSolid(numPrim, label); 
    AABB bb = {} ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        CSGPrim* pr = fd->addPrim(1, -1);  // numNode, nodeOffset
        CSGNode* nd = nullptr ; 
        switch(i)
        {
            case 0: nd = fd->addNode(CSGNode::Box3(2.f*halfside)) ;  break ;   
            case 1: nd = fd->addNode(CSGNode::Sphere(radius))     ;  break ;   
        } 
        pr->setAABB( nd->AABB() ); 
        bb.include_aabb( nd->AABB() ); 
    }

    so->center_extent = bb.center_extent() ;  
    LOG(info) << " so->center_extent " << so->center_extent ; 

    return so ; 
}



/**
CSGMaker::makeScaled
----------------------

Creates a CSGSolid composed of *layers* CSGPrim where each CSGPrim has one CSGNode
with different scale transforms created and associated with each CSGNode to make 
a Russian doll arrangement.  

This demonstrates adding a transform and associating it to a CSGNode.  

**/

CSGSolid* CSGMaker::makeScaled(const char* label, const char* demo_node_type, float outer_scale, unsigned layers )
{
    std::vector<float> scales ;
    for(unsigned i=0 ; i < layers ; i++) scales.push_back(outer_scale*float(layers-i)/float(layers)) ; 

    unsigned numPrim = layers ; 
    CSGSolid* so = fd->addSolid(numPrim, label); 
    AABB bb = {} ; 

    for(unsigned i=0 ; i < numPrim ; i++)
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1; 
        CSGPrim* pr = fd->addPrim(numNode, nodeOffset_); 
        CSGNode* nd = fd->addNode(CSGNode::MakeDemo(demo_node_type)) ;
    
        float scale = scales[i]; 
        const Tran<double>* tran_scale = Tran<double>::make_scale( double(scale), double(scale), double(scale) ); 

        /*
        unsigned transform_idx = 1 + fd->addTran(tran_scale);      // 1-based idx, 0 meaning None
        nd->setTransform(transform_idx); 
        const qat4* tr = fd->getTran(transform_idx-1u) ;   // storage uses 0-based 
        tr->transform_aabb_inplace( nd->AABB() ); 
        */

        bool transform_node_aabb = true ; 
        fd->addNodeTran( nd, tran_scale, transform_node_aabb ); 


        bb.include_aabb( nd->AABB() ); 

        // pr->setSbtIndexOffset(i) ;  //  NOW done in addPrim
        pr->setAABB( nd->AABB() ); 
    }

    so->center_extent = bb.center_extent() ;  
    LOG(info) << " so->center_extent " << so->center_extent ; 

    return so ; 
}



/**
CSGMaker::makeClustered
-------------------------


**/

CSGSolid* CSGMaker::makeClustered(const char* label,  int i0, int i1, int is, int j0, int j1, int js, int k0, int k1, int ks, double unit, bool inbox ) 
{
    unsigned numPrim = inbox ? 1 : 0 ; 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        //LOG(info) << std::setw(2) << numPrim << " (i,j,k) " << "(" << i << "," << j << "," << k << ") " ; 
        numPrim += 1 ; 
    }
       
    LOG(info) 
        << " label " << label  
        << " numPrim " << numPrim 
        << " inbox " << inbox
        ;  

    CSGSolid* so = fd->addSolid(numPrim, label);
    unsigned idx = 0 ; 

    AABB bb = {} ; 
 
    for(int i=i0 ; i < i1 ; i+=is ) 
    for(int j=j0 ; j < j1 ; j+=js ) 
    for(int k=k0 ; k < k1 ; k+=ks ) 
    {
        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ;  // -1:use current node count as about to add the declared numNode
        CSGPrim* p = fd->addPrim(numNode, nodeOffset_); 
        CSGNode* n = fd->addNode(CSGNode::MakeDemo(label)) ;
    
        const Tran<double>* translate = Tran<double>::make_translate( double(i)*unit, double(j)*unit, double(k)*unit ); 


        unsigned transform_idx = 1 + fd->addTran(translate);      // 1-based idx, 0 meaning None
        n->setTransform(transform_idx); 
        const qat4* t = fd->getTran(transform_idx-1u) ; 
        t->transform_aabb_inplace( n->AABB() ); 




        bb.include_aabb( n->AABB() ); 

        // p->setSbtIndexOffset(idx) ;    //  now done in addPrim
        p->setAABB( n->AABB() );  // HUH : THIS SHOULD BE bb ?

        //DumpAABB("p->AABB() aft setup", p->AABB() ); 
        
        LOG(info) << " idx " << idx << " transform_idx " << transform_idx ; 
 
        idx += 1 ; 
    }


    if(inbox)
    {
        float4 ce = bb.center_extent(); 
        float fullside = ce.w*2.f ; 

        unsigned numNode = 1 ; 
        int nodeOffset_ = -1 ;  // -1:use current node count as about to add the declared numNode

        CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 
        CSGNode bx = CSGNode::Box3(fullside) ;
        CSGNode* n = fd->addNode(bx); 


        const Tran<float>* to_center = Tran<float>::make_translate( float(ce.x), float(ce.y), float(ce.z) ); 
        unsigned transform_idx = 1 + fd->addTran(to_center);  // 1-based idx, 0 meaning None
        const qat4* t = fd->getTran(transform_idx-1u) ; 

        n->setTransform(transform_idx); 
        t->transform_aabb_inplace( n->AABB() ); 

        // p->setSbtIndexOffset(idx);   now done in addPrim
        p->setAABB( n->AABB() );
        
        idx += 1 ; 
    }


    so->center_extent = bb.center_extent() ;   // contains AABB of all CSGPrim 
    LOG(info) << " so->center_extent " << so->center_extent ; 
    return so ; 
}

/**
CSGMaker::makeSolid11 makes 1-CSGPrim with 1-CSGNode
---------------------------------------------------------
**/

CSGSolid* CSGMaker::makeSolid11(const char* label, CSGNode nd, const std::vector<float4>* pl, int meshIdx, const Tran<double>* tr  ) 
{
    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);

    unsigned numNode = 1 ; 
    int nodeOffset_ = -1 ;  
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_); 
    p->setMeshIdx(meshIdx); 

    CSGNode* n = fd->addNode(nd, pl, tr ); 
    p->setAABB( n->AABB() ); 

    float extent = p->extent(); 
    if(extent == 0.f )
        LOG(fatal) << "FATAL : " << label << " : got zero extent " ; 
    assert( extent > 0.f ); 

    AABB bb = AABB::Make( p->AABB() ); 
    so->center_extent = bb.center_extent()  ; 
    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}

CSGSolid* CSGMaker::makeBooleanBoxSphere( const char* label, unsigned op_, float radius, float fullside, int meshIdx )
{
    CSGNode bx = CSGNode::Box3(fullside) ; 
    CSGNode sp = CSGNode::Sphere(radius); 
    return makeBooleanTriplet(label, op_, bx, sp ); 
}


/**
CSGMaker::makeBooleanTriplet
------------------------------

Note convention that by-value CSGNode arguments are to be added to CSGFoundry
(ie assumed not already added) as opposed by pointer CSGNode arguments, which imply 
the nodes are already added to the CSGFoundry. 

          op

      left   right 
 
**/

CSGSolid* CSGMaker::makeBooleanTriplet( const char* label, unsigned op_, const CSGNode& left, const CSGNode& right, int meshIdx ) 
{
    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);

    unsigned numNode = 3 ; 
    int nodeOffset_ = -1 ; 
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 
    if(meshIdx > -1) p->setMeshIdx(meshIdx);

    CSGNode op = CSGNode::BooleanOperator(op_, 3); 
    CSGNode* n = fd->addNode(op); 

    fd->addNode(left); 
    fd->addNode(right); 
     
    // naive bbox combination yields overlarge bbox, not appropriate for production code
    AABB bb = {} ;
    bb.include_aabb( left.AABB() );     // assumes any transforms have been applied to the Node AABB
    bb.include_aabb( right.AABB() ); 
    p->setAABB( bb.data() );  

    so->center_extent = bb.center_extent()  ; 

    // setting transform as otherise loading foundry fails for lack of non-optional tran array 
    //const Tran<double>* tran_identity = Tran<double>::make_identity(); 
    unsigned transform_idx = 1 + fd->addTran();   // 1-based idx, 0 meaning None
    n->setTransform(transform_idx); 


    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}


CSGSolid* CSGMaker::makeOverlapBoxSphere( const char* label, float radius, float fullside )
{
    CSGNode bx = CSGNode::Box3(fullside) ; 
    CSGNode sp = CSGNode::Sphere(radius); 

    std::vector<CSGNode> leaves ; 
    leaves.push_back(bx); 
    leaves.push_back(sp); 

    return makeOverlapList( label, leaves, nullptr ); 
}

/**

                                    Y        

              (-side,side)          |             (side, side)
                         +          |             +
                                    |  
                                    |  
                                    |  
                                    |  
                                    |  
                      --------------O---------=---------  X
                                    |  
                                    |  
                                    |  
                                    |  
                                    |  
                                    |  
                                    +
                                    | ( 0, -side*sqrt(2))
                                    |  


               

**/
CSGSolid* CSGMaker::makeListThreeSphere( const char* label, unsigned type, float radius, float side )
{
    CSGNode s0 = CSGNode::Sphere(radius); 
    CSGNode s1 = CSGNode::Sphere(radius); 
    CSGNode s2 = CSGNode::Sphere(radius); 

    const Tran<double>* t0 = Tran<double>::make_translate(  side,  side         , 0. ); 
    const Tran<double>* t1 = Tran<double>::make_translate( -side,  side         , 0. ); 
    const Tran<double>* t2 = Tran<double>::make_translate(    0., -side*sqrt(2.), 0. ); 


    std::vector<CSGNode> leaves ; 
    leaves.push_back(s0); 
    leaves.push_back(s1); 
    leaves.push_back(s2); 

    std::vector<const Tran<double>*> tran ; 
    tran.push_back(t0); 
    tran.push_back(t1); 
    tran.push_back(t2); 

    return makeList( label, type, leaves, &tran ); 
}

CSGSolid* CSGMaker::makeOverlapThreeSphere( const char* label, float radius, float side )
{
    return makeListThreeSphere( label, CSG_OVERLAP , radius, side ); 
}
/**
CSGMaker::makeContiguousThreeSphere
---------------------------------------
radius 100.f side 50.f 


    (-side,side,0)          (side,side,0)    
    (-50,50,0)              (50,50,0) 
        s1         |         s0 
          +        |         +
                   |
                   |
          ---------O-----------
                   |
                   |
                   +
                  s2
                (0,-70,0)
               (0,-side*sqrt(2),0)


TIP: for debugging temporarily switch to CSG_DISCONTIGUOUS in order to see the separate spheres
**/

CSGSolid* CSGMaker::makeContiguousThreeSphere( const char* label, float radius, float side )
{
    return makeListThreeSphere( label, CSG_CONTIGUOUS , radius, side ); 
}

/**
CSGMaker::makeDiscontiguousThreeSphere
-----------------------------------------

radius 100.f side 100.f 

    (-side,side,0)          (side,side,0)    
    (-100,100,0)            (100,100,0) 
     s1            |            s0 
      +            |             +
                   |
                   |
                   |
                   |
                   |
          ---------O-----------
                   |
                   |
                   |
                   |
                   |
                   |
                   +
                  s2
                (0,-170,0)
               (0,-side*sqrt(2),0)


**/

CSGSolid* CSGMaker::makeDiscontiguousThreeSphere( const char* label, float radius, float side )
{
    return makeListThreeSphere( label, CSG_DISCONTIGUOUS , radius, side ); 
}


/**
CSGMaker::makeDiscontiguousTwoSphere
-------------------------------------

Checking the sense of the transforms : get the expected ones. 


                   Y
    (-100,200)     |
    (-side, 2*side)|
         t1        |
           +       |
                   |
                   |
                   |      +   t0 (side, 0.5*side)    (100,50)
                   |                            
                   |
        -----------O------------------------------ X

**/

CSGSolid* CSGMaker::makeDiscontiguousTwoSphere( const char* label, float radius, float side )
{
    return makeListTwoSphere( label, CSG_DISCONTIGUOUS , radius, side ); 
}

CSGSolid* CSGMaker::makeListTwoSphere( const char* label, unsigned type, float radius, float side )
{
    CSGNode s0 = CSGNode::Sphere(radius); 
    CSGNode s1 = CSGNode::Sphere(radius); 

    const Tran<double>* t0 = Tran<double>::make_translate(  side,  0.5*side     , 0. ); 
    const Tran<double>* t1 = Tran<double>::make_translate( -side,  2.0*side      , 0. ); 

    std::vector<CSGNode> leaves ; 
    leaves.push_back(s0); 
    leaves.push_back(s1); 

    std::vector<const Tran<double>*> tran ; 
    tran.push_back(t0); 
    tran.push_back(t1); 
 
    return makeList( label, type, leaves, &tran ); 
}




CSGSolid* CSGMaker::makeContiguousBoxSphere( const char* label, float radius, float fullside )
{
    CSGNode bx = CSGNode::Box3(fullside) ; 
    CSGNode sp = CSGNode::Sphere(radius); 

    std::vector<CSGNode> leaves ; 
    leaves.push_back(bx); 
    leaves.push_back(sp); 

    return makeContiguousList( label, leaves, nullptr ); 
}

CSGSolid* CSGMaker::makeDiscontiguousBoxSphere( const char* label, float radius, float fullside )
{
    CSGNode bx = CSGNode::Box3(fullside) ; 
    CSGNode sp = CSGNode::Sphere(radius); 

    const Tran<double>* bx_shift = Tran<double>::make_translate( fullside, 0., 0. ) ;  
    const Tran<double>* sp_shift = Tran<double>::make_translate(-fullside, 0., 0. ) ;  

    std::vector<CSGNode> leaves ; 
    leaves.push_back(bx); 
    leaves.push_back(sp); 

    std::vector<const Tran<double>*> trans ;
    trans.push_back(bx_shift); 
    trans.push_back(sp_shift); 

    return makeDiscontiguousList( label, leaves, &trans ); 
}




CSGSolid* CSGMaker::makeOverlapList(       const char* label, std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* tran )
{  
    return makeList( label, CSG_OVERLAP, leaves, tran ); 
}
CSGSolid* CSGMaker::makeContiguousList(    const char* label, std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* tran  )
{ 
    return makeList( label, CSG_CONTIGUOUS, leaves, tran ); 
}
CSGSolid* CSGMaker::makeDiscontiguousList( const char* label, std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* tran  )
{ 
    return makeList( label, CSG_DISCONTIGUOUS, leaves, tran ); 
}


CSGSolid* CSGMaker::makeList( const char* label, unsigned type, std::vector<CSGNode>& leaves, const std::vector<const Tran<double>*>* tran )
{
    unsigned numSub = leaves.size() ; 
    unsigned numTran = tran ? tran->size() : 0  ; 
    if( numTran > 0 ) assert( numSub == numTran );
 
    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);
    
    unsigned numNode = 1 + numSub ; 
    int nodeOffset_ = -1 ; 
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 

    unsigned subOffset = 1 ; // now using absolute offsets from "root" to the first sub  see notes/issues/ContiguousThreeSphere.rst
    CSGNode hdr = CSGNode::ListHeader(type, numSub, subOffset ); 
    CSGNode* n = fd->addNode(hdr); 

    AABB bb = {} ;
    fd->addNodes( bb, leaves, tran ); 
    p->setAABB( bb.data() );  
    so->center_extent = bb.center_extent()  ; 

    fd->addNodeTran(n);   // setting identity transform 
    
    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}


/**
CSGMaker::makeUnionListBoxSphere
---------------------------------

This is for testing a tree with a compound node within it  


          op
         /  \
        bx  (co)  
             |
             sp 


List notes require subNum and subOffset to identify the referenced sequence of sub-nodes. 
The subOffset in a tree with only a single list will simply be the number of tree nodes 
which is 3 in this simple example. 
For larger trees it will be the number of nodes in the complete binary tree 
which will be: 3, 7, 15, 31  

When more than one list the subOffset will need to be arranged to 
skip over the subs of other lists. 

**/

CSGSolid* CSGMaker::makeUnionListBoxSphere( const char* label, float radius, float fullside )
{
    // 3 tree nodes + 1 sub-node from the compound
    CSGNode op    = CSGNode::BooleanOperator(CSG_UNION, 3); 
    CSGNode left  = CSGNode::Box3(fullside) ; 

    int subNum = 1 ;    // number of subs referenced by the List node
    int subOffset = op.subNum() ; 
    CSGNode right = CSGNode::ListHeader(CSG_CONTIGUOUS, subNum, subOffset ); 
    CSGNode sub   = CSGNode::Sphere(radius); 

    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);

    unsigned numNode = 3 + subNum ; 
    int nodeOffset_ = -1 ; 
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 

    CSGNode* root = fd->addNode(op);   
    CSGNode* l    = fd->addNode(left); 
    CSGNode* r    = fd->addNode(right); 
    CSGNode* s    = fd->addNode(sub); 

    assert( op.subNum() == 3 && root->subNum() == 3 ); 
     
    assert( root->typecode() == CSG_UNION ); 
    assert( l->typecode() == CSG_BOX3 ); 
    assert( r->typecode() == CSG_CONTIGUOUS ); 
    assert( s->typecode() == CSG_SPHERE ); 

    AABB bb = {} ;
    bb.include_aabb( right.AABB() ); 
    bb.include_aabb( sub.AABB() ); 
    p->setAABB( bb.data() );  
    so->center_extent = bb.center_extent()  ; 

    fd->addNodeTran(root);  // adding identity, as geometry must have at least one transform

    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}


/**
CSGMaker::makeBooleanListList
-------------------------------

This generalizes from CSGMaker::makeUnionListBoxSphere in order to test in a more flexible way.

TODO: need to apply the experience from here to the GeoChain conversions with CSG_CONTIGUOUS hinting, 
should the subOffset be set at NCSG level ?

**/

CSGSolid* CSGMaker::makeBooleanListList( const char* label, 
       unsigned op_, 
       unsigned ltype,
       unsigned rtype,  
       std::vector<CSGNode>& lhs, 
       std::vector<CSGNode>& rhs, 
       const std::vector<const Tran<double>*>* ltran,
       const std::vector<const Tran<double>*>* rtran
    )
{
    unsigned num_left  = lhs.size(); 
    unsigned num_right = rhs.size(); 
    assert( num_left > 0 && num_right > 0 ); 

    // singles on left or right are inlined into the boolean so no addition beyond the tree
    unsigned numNode = 3 + ( num_left == 1 ? 0 : num_left ) + ( num_right == 1 ? 0 : num_right )  ; 

    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);

    int nodeOffset_ = -1 ; 
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 

    CSGNode op    = CSGNode::BooleanOperator(op_, 3); 
    CSGNode* root = fd->addNode(op); 

    unsigned subOffset = 0 ; 
    subOffset += root->subNum() ;  
    assert( subOffset == 3 );   // 3 tree nodes


    AABB bb = {} ;

    if( num_left == 1 && num_right == 1 )
    {
        const CSGNode& left = lhs[0]; 
        const CSGNode& right = rhs[0]; 

        fd->addNode(bb, left); 
        fd->addNode(bb, right); 
    }
    else if( num_left > 1 && num_right == 1 )
    {
        CSGNode left = CSGNode::ListHeader(ltype, num_left, subOffset ); 
        subOffset += num_left ; 

        const CSGNode& right = rhs[0]; 

        fd->addNode(left); 
        fd->addNode(bb, right); 
        fd->addNodes(bb, lhs, ltran); 
    }
    else if( num_left == 1 && num_right > 1 )
    {
        CSGNode left = lhs[0] ; 
        CSGNode right = CSGNode::ListHeader(rtype, num_right, subOffset ); 
        subOffset += num_right ; 

        fd->addNode(bb, left); 
        fd->addNode(right); 
        fd->addNodes(bb, rhs, rtran); 
    }
    else if( num_left > 1 && num_right > 1 )
    {
        CSGNode left = CSGNode::ListHeader(ltype, num_left, subOffset ); 
        subOffset += num_left ; 

        CSGNode right = CSGNode::ListHeader(rtype, num_right, subOffset ); 
        subOffset += num_right ; 

        fd->addNode(left); 
        fd->addNode(right); 

        fd->addNodes( bb, lhs, ltran ); 
        fd->addNodes( bb, rhs, rtran ); 
    }


    p->setAABB( bb.data() );  
    so->center_extent = bb.center_extent()  ; 

    fd->addNodeTran(root);  // adding identity, as geometry must have at least one transform
    
    return so ; 
}


/**
CSGMaker::makeUnionLLBoxSphere
--------------------------------

radius=100.f fullside=100.f


                                   (-50,100)                  (50,100)
                                    +                           +
                          .                 .             .            .
                                                   +  
                     .                                                       .
                                              .           . 
 
                  .                                                                 .
                                         .

              .                       .                       \                        .
             
                               
           -|-----------------------s0-------------O------------|s1--------------------|-------------
                              (-50,0,0)                       (50,0,0)                        

              \                        \                      /                       /


                   
                 
                                                    +


                                    +                           +
                                  (-50,-100)                 (50, -100)


            |          |            |             |            |           |          |
          -150        -100         -50            0            50         100        150
          
**/

CSGSolid* CSGMaker::makeUnionLLBoxSphere( const char* label, float radius, float fullside  )
{
    std::vector<CSGNode> lhs ; 
    lhs.push_back( CSGNode::Sphere(radius) ); 
    lhs.push_back( CSGNode::Sphere(radius) ); 

    std::vector<const Tran<double>*> ltran ;
    ltran.push_back(Tran<double>::make_translate( -radius/2., 0., 0. )) ;
    ltran.push_back(Tran<double>::make_translate(  radius/2., 0., 0. )) ;

    std::vector<CSGNode> rhs ; 
    rhs.push_back( CSGNode::Box3(fullside) ); 
    rhs.push_back( CSGNode::Box3(fullside) ); 

    std::vector<const Tran<double>*> rtran ;
    rtran.push_back(Tran<double>::make_translate(   0.,  fullside, 0. )) ;
    rtran.push_back(Tran<double>::make_translate(   0., -fullside, 0. )) ;

    unsigned btype = CSG_DIFFERENCE ; 
    unsigned ltype = CSG_CONTIGUOUS ; 
    unsigned rtype = CSG_DISCONTIGUOUS  ;   

    return makeBooleanListList(label, btype, ltype, rtype, lhs, rhs, &ltran, &rtran ); 
}

CSGSolid* CSGMaker::makeListTwoBoxTwoSphere( const char* label, float radius, float fullside  )
{
    std::vector<CSGNode> leaves ; 
    leaves.push_back( CSGNode::Sphere(radius) ); 
    leaves.push_back( CSGNode::Sphere(radius) ); 
    leaves.push_back( CSGNode::Box3(fullside) ); 
    leaves.push_back( CSGNode::Box3(fullside) ); 

    std::vector<const Tran<double>*> tran ;
    tran.push_back(Tran<double>::make_translate( -radius/2., 0., 0. )) ;
    tran.push_back(Tran<double>::make_translate(  radius/2., 0., 0. )) ;
    tran.push_back(Tran<double>::make_translate(   0.,  fullside, 0. )) ;
    tran.push_back(Tran<double>::make_translate(   0., -fullside, 0. )) ;

    unsigned type = CSG_DISCONTIGUOUS ; // useful during dev to see constituents
    //unsigned type = CSG_CONTIGUOUS ; 
    return makeList( label, type, leaves, &tran ); 
}


/**
CSGMaker::makeBooleanSeptuplet
----------------------------------


                        1:t

             10:l                   11:r
 
       100:ll    101:lr       110:rl     111:rr

**/

CSGSolid* CSGMaker::makeBooleanSeptuplet( 
    const char* label, 
    const CSGNode& top, 
    const CSGNode& l, 
    const CSGNode& r, 
    const CSGNode& ll, 
    const CSGNode& lr, 
    const CSGNode& rl, 
    const CSGNode& rr, 
    const int meshIdx ) 
{
    unsigned numPrim = 1 ; 
    CSGSolid* so = fd->addSolid(numPrim, label);

    unsigned numNode = 7 ; 
    int nodeOffset_ = -1 ; 
    CSGPrim* p = fd->addPrim(numNode, nodeOffset_ ); 
    if(meshIdx > -1) p->setMeshIdx(meshIdx); 

    std::vector<CSGNode> nn = {top, l, r, ll, lr, rl, rr } ; 
    assert( nn.size() == numNode ); 

    CSGNode* tptr = nullptr ; 
    AABB bb = {} ;
    for(unsigned i=0 ; i < numNode ; i++ )
    {
        const CSGNode& n = nn[i] ; 
        CSGNode* nptr = fd->addNode(n); 
        if(i == 0) tptr = nptr ;  

        if( n.is_primitive() && !n.is_zero() && !n.is_complement() )
        {
            bb.include_aabb( n.AABB() );   // naive bbox combination : overlarge bbox
        }
    }
    p->setAABB( bb.data() );  

    so->center_extent = bb.center_extent()  ; 

    // setting transform as otherise loading foundry fails for lack of non-optional tran array 
    //const Tran<double>* tran_identity = Tran<double>::make_identity(); 
    unsigned transform_idx = 1 + fd->addTran();   // 1-based idx, 0 meaning None
    tptr->setTransform(transform_idx); 

    LOG(info) << "so.label " << so->label << " so.center_extent " << so->center_extent ; 
    return so ; 
}


CSGSolid* CSGMaker::makeDifferenceCylinder( const char* label, float rmax, float rmin, float z1, float z2, float z_inner_factor   )
{
    assert( rmax > rmin ); 
    assert( z2 > z1 ); 

    float px = 0.f ;  
    float py = 0.f ;  

    CSGNode outer = CSGNode::Cylinder( 0.f, 0.f, rmax, z1, z2 ); 
    CSGNode inner = CSGNode::Cylinder( px, py,   rmin, z1*z_inner_factor, z2*z_inner_factor ); 
    return makeBooleanTriplet(label, CSG_DIFFERENCE, outer, inner ); 
}

/**
CSGMaker::makeBoxSubSubCylinder
---------------------------------

         
                   t:di
          
          l:bx             r:di 
 
       ll:ze  lr:ze    rl:cy    rr:cy
              
 
**/


CSGSolid* CSGMaker::makeBoxSubSubCylinder( const char* label, float fullside, float rmax, float rmin, float z1, float z2, float z_inner_factor   )
{
    CSGNode t = CSGNode::BooleanOperator(CSG_DIFFERENCE, -1); 
    CSGNode l = CSGNode::Box3(fullside) ;
    CSGNode r = CSGNode::BooleanOperator(CSG_DIFFERENCE, -1); 
    CSGNode ll = CSGNode::Zero(); 
    CSGNode lr = CSGNode::Zero(); 
    CSGNode rl = CSGNode::Cylinder( 0.f, 0.f, rmax, z1, z2 ); 
    CSGNode rr = CSGNode::Cylinder( 0.f, 0.f, rmin, z1*z_inner_factor, z2*z_inner_factor ); 
    int meshIdx = -1 ;  
    return makeBooleanSeptuplet(label, t, l, r, ll, lr, rl, rr, meshIdx );  
}





CSGSolid* CSGMaker::makeUnionBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, CSG_UNION, radius, fullside, UBSP_MIDX ); 
}
CSGSolid* CSGMaker::makeIntersectionBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, CSG_INTERSECTION, radius, fullside, IBSP_MIDX ); 
}
CSGSolid* CSGMaker::makeDifferenceBoxSphere( const char* label, float radius, float fullside ){
    return makeBooleanBoxSphere(label, CSG_DIFFERENCE, radius, fullside, DBSP_MIDX ); 
}





CSGSolid* CSGMaker::makeSphere(const char* label, float radius)
{
    CSGNode nd = CSGNode::Sphere(radius); 
    return makeSolid11(label, nd, nullptr, SPHE_MIDX ); 
}

/**
CSGMaker::makeEllipsoid
--------------------------

Hmm adding the transform to CSGFoundry before the prim gets added is problematic
for CSGCloneTest as transfrom offsets get captured at prim creation. 

**/

CSGSolid* CSGMaker::makeEllipsoid(  const char* label, float rx, float ry, float rz )
{
    CSGNode nd = CSGNode::Sphere(rx);

    double dx = double(rx) ; 
    double dy = double(ry) ; 
    double dz = double(rz) ; 
 
    double sx = double(1.) ; 
    double sy = dy/dx ; 
    double sz = dz/dx ; 

    const Tran<double>* tr = Tran<double>::make_scale(sx, sy, sz ); 

    return makeSolid11(label, nd, nullptr, ELLI_MIDX, tr ); 
}


CSGSolid* CSGMaker::makeRotatedCylinder(const char* label, float px, float py, float radius, float z1, float z2, float ax, float ay, float az, float angle_deg )
{
    CSGNode nd = CSGNode::Cylinder( px, py, radius, z1, z2 ); 
    const Tran<double>* tr = Tran<double>::make_rotate(ax, ay, az, angle_deg ); 
    return makeSolid11(label, nd, nullptr, RCYL_MIDX, tr  ); 
}


CSGSolid* CSGMaker::makeInfCylinder(const char* label, float radius, float hz )
{
    CSGNode nd = CSGNode::InfCylinder( radius, hz ); 
    return makeSolid11(label, nd, nullptr, ICYL_MIDX ); 
}

CSGSolid* CSGMaker::makeInfPhiCut(const char* label, float startPhi_pi, float deltaPhi_pi )
{
    CSGNode nd = CSGNode::InfPhiCut(startPhi_pi, deltaPhi_pi ); 
    return makeSolid11(label, nd, nullptr, IPHI_MIDX ); 
}


CSGSolid* CSGMaker::makeInfThetaCut(const char* label, float startTheta_pi, float deltaTheta_pi )
{
    CSGNode nd = CSGNode::InfThetaCut(startTheta_pi, deltaTheta_pi ); 
    return makeSolid11(label, nd, nullptr, ITHE_MIDX ); 
}
CSGSolid* CSGMaker::makeInfThetaCutL(const char* label, float startTheta_pi, float deltaTheta_pi )
{
    CSGNode nd = CSGNode::InfThetaCut(startTheta_pi, deltaTheta_pi ); 
    return makeSolid11(label, nd, nullptr, ITHL_MIDX ); 
}





CSGSolid* CSGMaker::makeZSphere(const char* label, float radius, float z1, float z2)
{
    CSGNode nd = CSGNode::ZSphere(radius, z1, z2); 
    return makeSolid11(label, nd, nullptr, ZSPH_MIDX ); 
}

CSGSolid* CSGMaker::makeCone(const char* label, float r1, float z1, float r2, float z2)
{
    CSGNode nd = CSGNode::Cone(r1, z1, r2, z2 ); 
    return makeSolid11(label, nd, nullptr, CONE_MIDX ); 
}

CSGSolid* CSGMaker::makeHyperboloid(const char* label, float r0, float zf, float z1, float z2)
{
    CSGNode nd = CSGNode::Hyperboloid( r0, zf, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, HYPE_MIDX ); 
}

CSGSolid* CSGMaker::makeBox3(const char* label, float fx, float fy, float fz )
{
    CSGNode nd = CSGNode::Box3(fx, fy, fz); 
    return makeSolid11(label, nd, nullptr, BOX3_MIDX ); 
}

CSGSolid* CSGMaker::makePlane(const char* label, float nx, float ny, float nz, float d)
{
    CSGNode nd = CSGNode::Plane(nx, ny, nz, d ); 
    return makeSolid11(label, nd, nullptr, PLAN_MIDX ); 
}

CSGSolid* CSGMaker::makeSlab(const char* label, float nx, float ny, float nz, float d1, float d2 )
{
    CSGNode nd = CSGNode::Slab( nx, ny, nz, d1, d1 ); 
    return makeSolid11(label, nd, nullptr, SLAB_MIDX ); 
}

CSGSolid* CSGMaker::makeCylinder(const char* label, float px, float py, float radius, float z1, float z2)
{
    CSGNode nd = CSGNode::Cylinder( px, py, radius, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, CYLI_MIDX ); 
}

CSGSolid* CSGMaker::makeDisc(const char* label, float px, float py, float ir, float r, float z1, float z2)
{
    CSGNode nd = CSGNode::Disc(px, py, ir, r, z1, z2 ); 
    return makeSolid11(label, nd, nullptr, DISC_MIDX ); 
}


float4 CSGMaker::TriPlane( const std::vector<float3>& v, unsigned i, unsigned j, unsigned k )  // static 
{
    // normal for plane through v[i] v[j] v[k]
    float3 ij = v[j] - v[i] ; 
    float3 ik = v[k] - v[i] ; 
    float3 n = normalize(cross(ij, ik )) ;
    float di = dot( n, v[i] ) ;
    //float dj = dot( n, v[j] ) ;
    //float dk = dot( n, v[k] ) ;
    //LOG(info) << " di " << di << " dj " << dj << " dk " << dk << " n (" << n.x << "," << n.y << "," << n.z << ")" ; 
    float4 plane = make_float4( n, di ) ; 
    return plane ;  
}

CSGSolid* CSGMaker::makeConvexPolyhedronCube(const char* label, float extent)
{
    float hx = extent ; 
    float hy = extent/2.f ; 
    float hz = extent/3.f ; 

    std::vector<float4> pl ; 
    pl.push_back( make_float4(  1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4( -1.f,  0.f,  0.f, hx ) ); 
    pl.push_back( make_float4(  0.f,  1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f, -1.f,  0.f, hy ) ); 
    pl.push_back( make_float4(  0.f,  0.f,  1.f, hz ) ); 
    pl.push_back( make_float4(  0.f,  0.f, -1.f, hz ) );

    CSGNode nd = {} ;
    nd.setAABB(-hx, -hy, -hz, hx, hy, hz); 
    return makeSolid11(label, nd, &pl, VCUB_MIDX ); 
}


/*  
     https://en.wikipedia.org/wiki/Tetrahedron

       0:(1,1,1)
       1:(1,−1,−1)
       2:(−1,1,−1) 
       3:(−1,−1,1)

                              (1,1,1)
                 +-----------0
                /|          /| 
     (-1,-1,1) / |         / |
              3-----------+  |
              |  |        |  |
              |  |        |  |
   (-1,1,-1)..|..2--------|--+
              | /         | /
              |/          |/
              +-----------1
                          (1,-1,-1)      

          Faces (right-hand-rule oriented outwards normals)
                0-1-2
                1-3-2
                3-0-2
                0-3-1

         z  y
         | /
         |/
         +---> x
*/

CSGSolid* CSGMaker::makeConvexPolyhedronTetrahedron(const char* label, float extent)
{
    //extent = 100.f*sqrt(3); 
    float s = extent ; 

    std::vector<float3> vtx ; 
    vtx.push_back(make_float3( s, s, s));  
    vtx.push_back(make_float3( s,-s,-s)); 
    vtx.push_back(make_float3(-s, s,-s)); 
    vtx.push_back(make_float3(-s,-s, s)); 

    std::vector<float4> pl ; 
    pl.push_back(TriPlane(vtx, 0, 1, 2)) ;  
    pl.push_back(TriPlane(vtx, 1, 3, 2)) ;  
    pl.push_back(TriPlane(vtx, 3, 0, 2)) ;  
    pl.push_back(TriPlane(vtx, 0, 3, 1)) ;  

    //for(unsigned i=0 ; i < pl.size() ; i++) LOG(info) << " pl (" << pl[i].x << "," << pl[i].y << "," << pl[i].z << "," << pl[i].w << ") " ;
 
    CSGNode nd = {} ;
    nd.setAABB(extent); 
    return makeSolid11(label, nd, &pl, VTET_MIDX ); 
}


