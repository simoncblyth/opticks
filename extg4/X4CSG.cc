#include <iostream>
#include <cstring>

#include "PLOG.hh"
#include "SSys.hh"
#include "BStr.hh"
#include "BFile.hh"

#include "NTreeAnalyse.hpp"
#include "NTreeProcess.hpp"
#include "NNode.hpp"
#include "NCSG.hpp"
#include "NPYBase.hpp"
#include "NPYMeta.hpp"
#include "NCSGData.hpp"
#include "NCSGList.hpp"

#include "X4Solid.hh"
#include "X4CSG.hh"

#include "G4Box.hh"
#include "G4VSolid.hh"

//#include "G4VisExtent.hh"
#include "X4SolidExtent.hh"


void X4CSG::Serialize( const G4VSolid* solid, const char* csgpath ) // static
{
    X4CSG xcsg(solid);
    std::cerr << xcsg.save(csgpath) << std::endl ;   // NB only stderr emission to be captured by bash 
    xcsg.dumpTestMain(); 
}

const char* X4CSG::GenerateTestPath( const char* prefix, unsigned lvidx ) // static
{ 
    std::string dir = BFile::FormPath( prefix, "tests" ); 
    std::string name = BStr::concat("x", BStr::utoa(lvidx, 3, true), ".cc") ; 
    bool create = true ; 
    std::string path = BFile::preparePath(dir.c_str(), name.c_str(), create); 
    return strdup(path.c_str()); 
}

void X4CSG::GenerateTest( const G4VSolid* solid, const char* prefix, unsigned lvidx )  // static
{
    const char* path = GenerateTestPath(prefix, lvidx) ; 
    LOG(debug) << "( " << lvidx << " " << path ; 
    X4CSG xcsg(solid);
    xcsg.writeTestMain(path); 
    LOG(debug) << ") " << lvidx ; 
}

G4VSolid* X4CSG::MakeContainer(const G4VSolid* solid, float scale ) // static
{
   /*
    G4VisExtent ve = solid->GetExtent();  // crashes in 10.4.2
    float xmin = ve.GetXmin() ;
    float ymin = ve.GetYmin() ;
    float zmin = ve.GetZmin() ;

    float xmax = ve.GetXmax() ;
    float ymax = ve.GetYmax() ;
    float zmax = ve.GetZmax() ;
    */

    nbbox* bb = X4SolidExtent::Extent(solid) ; 

    float xmin = bb->min.x ; 
    float ymin = bb->min.y ; 
    float zmin = bb->min.z ; 

    float xmax = bb->max.x ; 
    float ymax = bb->max.y ; 
    float zmax = bb->max.z ; 

    float hx0 = std::max(std::abs(xmin),std::abs(xmax)) ; 
    float hy0 = std::max(std::abs(ymin),std::abs(ymax)) ; 
    float hz0 = std::max(std::abs(zmin),std::abs(zmax)) ; 

    float hh = scale*std::max(std::max( hx0, hy0 ), hz0 ); 

    float hx = hh ; 
    float hy = hh ; 
    float hz = hh ;    // equi-dimensioned container easier to navigate with  

    G4VSolid* container = new G4Box("container", hx, hy, hz ); 

    return container ; 
}

std::string X4CSG::desc() const
{
    return "X4CSG" ; 
}

X4CSG::X4CSG(const G4VSolid* solid_)
    :
    verbosity(SSys::getenvint("VERBOSITY",0)),
    solid(solid_),
    container(MakeContainer(solid, 1.5f)),
    solid_boundary("Vacuum///GlassSchottF2"),
    container_boundary("Rock//perfectAbsorbSurface/Vacuum"),
    nraw(X4Solid::Convert(solid, solid_boundary)),
    nsolid(X4Solid::Balance(nraw)),
    ncontainer(X4Solid::Convert(container, container_boundary)),
    csolid( NCSG::Adopt(nsolid) ),
    ccontainer( NCSG::Adopt(ncontainer) ),
    ls(NULL)
{
    init();
}

void X4CSG::init()
{
    //checkTree();  

    configure( csolid->getMeta() ) ; 
    configure( ccontainer->getMeta() ) ; 

    trees.push_back( ccontainer ); 
    trees.push_back( csolid ); 
}


void X4CSG::checkTree() const 
{
    unsigned soIdx = 0 ; 
    unsigned lvIdx = 999 ;  // a listed value 

    //LOG(info) << " nraw " << std::endl << NTreeAnalyse<nnode>::Desc(nraw) ;

    nnode* pro = NTreeProcess<nnode>::Process(nraw, soIdx, lvIdx);  // balances deep trees 
    assert( pro ) ; 

    //LOG(info) << " pro " << std::endl << NTreeAnalyse<nnode>::Desc(pro) ;
}


void X4CSG::configure( NPYMeta* meta )
{
    meta->setValue<std::string>( "poly", "IM" );  
    meta->setValue<std::string>( "resolution", "20" );  
}

void X4CSG::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::cout << "solid" << std::endl << *solid << std::endl ; 
    std::cout << "container" << std::endl << *container << std::endl ; 
}
 
std::string X4CSG::configuration(const char* csgpath) const 
{
    std::stringstream ss ; 
    ss << "analytic=1_csgpath=" << csgpath ;    // TODO: not convenient to hardcode analytic=1 here 
    return ss.str(); 
}

std::string X4CSG::save(const char* csgpath) 
{
    ls = NCSGList::Create( trees, csgpath , verbosity ); 
    ls->savesrc(); 
    return configuration(csgpath); 
}

void X4CSG::loadcheck(const char* csgpath) const 
{
     NCSGList* ls2 = NCSGList::Load( csgpath , verbosity );   // see whats missing from the save 
     assert( ls2 ) ; 
}


const std::string X4CSG::HEAD = R"(

#include "OPTICKS_LOG.hh"
#include "BFile.hh"
#include "X4.hh"
#include "X4CSG.hh"

#include "G4Box.hh"
#include "G4Orb.hh"
#include "G4Tubs.hh"
#include "G4Sphere.hh"
#include "G4Trd.hh"
#include "G4Polycone.hh"
#include "G4Cons.hh"
#include "G4Ellipsoid.hh"
#include "G4Torus.hh"

#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"
#include "G4SubtractionSolid.hh"

#include "G4RotationMatrix.hh"
#include "G4ThreeVector.hh"


)" ; 

const std::string X4CSG::TAIL = R"(

int main( int argc , char** argv )
{
    OPTICKS_LOG(argc, argv);

    const char* exename = PLOG::instance->args.exename() ; 

    G4VSolid* solid = make_solid() ; 

    std::string csgpath = BFile::FormPath(X4::X4GEN_DIR, exename) ; 

    X4CSG::Serialize( solid, csgpath.c_str() ) ;

    return 0 ; 
}
)" ; 


void X4CSG::generateTestMain( std::ostream& out ) const 
{
    if( nsolid->g4code == NULL ) 
    { 
        LOG(error) << " skip as no g4code " ; 
        return ;
    }   

    out << HEAD ; 
    nnode::to_g4code(nsolid, out,  0 ) ;  
    out << TAIL ; 
}
void X4CSG::dumpTestMain(const char* msg) const 
{
    LOG(info) << msg ; 
    std::ostream& out = std::cout ;
    generateTestMain( out ); 
}
void X4CSG::writeTestMain( const char* path_ ) const 
{
    std::string path = BFile::FormPath(path_) ;
    std::ofstream out(path.c_str());
    generateTestMain( out ); 
}



