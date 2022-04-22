
#include "scuda.h"
#include "SStr.hh"
#include "SSys.hh"
#include "SPath.hh"
#include "NP.hh"
#include "SOpticksResource.hh"

#include "QBnd.hh"
#include "QTex.hh"
#include "OPTICKS_LOG.hh"


void test_descBoundary(QBnd& qb)
{
    unsigned num_boundary = qb.getNumBoundary(); 
    LOG(info) 
        << " num_boundary " << num_boundary 
        << " qb.descBoundary " 
        << std::endl 
        << qb.descBoundary()
        ;
}


void test_getBoundarySpec(const QBnd& qb)
{
    unsigned num_boundary = qb.getNumBoundary(); 
    LOG(info) 
        << " num_boundary " << num_boundary 
        ;
    for(unsigned i=0 ; i < num_boundary ; i++)
    {
       std::cout 
           << std::setw(4) << i 
           << " : "
           << qb.getBoundarySpec(i)
           << std::endl 
           ;
    }
}

void test_getBoundaryLine(QBnd& qb)
{
    const char* spec = SSys::getenvvar("QCTX_SPEC", "Acrylic///LS" ); 
    unsigned idx = qb.getBoundaryIndex(spec); 
    if( idx == QBnd::MISSING ) 
    {
        LOG(error) << " QBnd MISSING spec " << spec ; 
        return ; 
    }

    unsigned num_boundary = qb.getNumBoundary(); 

    enum { IMAT = 3 } ;  
    unsigned line = qb.getBoundaryLine(spec, IMAT); 
    unsigned xline = idx*4 + IMAT ; 
    LOG(info)
        << " spec " << spec 
        << " idx " << idx  
        << " line " << line  
        << " xline " << xline  
        ; 

    assert( xline == line ); 

    unsigned line_max = (num_boundary-1)*4 + IMAT ; 
    unsigned linek_max = 2*line_max + 1 ;  

    LOG(info)
        << " line_max " << line_max 
        << " linek_max " << linek_max
        << " linek_max+1 " << linek_max+1
        << " qb.tex->height " << qb.tex->height
        << " qb.tex->width " << qb.tex->width
        ;

    assert( linek_max + 1 == qb.tex->height ); 
}

void test_getMaterialLine(QBnd& qb)
{
    std::vector<std::string> materials ; 
    SStr::Split( SSys::getenvvar("QCTX_MATERIALS", "Water,LS,Pyrex,Acrylic,NonExisting" ), ',', materials ); 
    LOG(info) << " materials.size " << materials.size() ; 

    for(unsigned i=0 ; i < materials.size() ; i++)
    { 
        const char* material = materials[i].c_str() ; 
        unsigned line = qb.getMaterialLine(material); 
        std::cout 
            << " material " << std::setw(50) << material
            << " line " << line
            << std::endl 
            ;
    }
}


/**
test_lookup_technical
----------------------

Technical test doing lookups over the entire texture.
TODO: a test more like actual usage.

**/

void test_lookup_technical(QBnd& qb)
{
    NP* lookup = qb.lookup(); 
    const char* dir = SPath::Resolve("$TMP/QBndTest", DIRPATH) ; 
    LOG(info) << " save to " << dir  ; 
    lookup->save(dir, "dst.npy"); 
    qb.src->save(dir, "src.npy") ; 
}


void test_getBoundaryIndices(const QBnd& qb)
{
    const char* bnd_fallback = "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum" ;  
    const char* bnd_sequence = SSys::getenvvar("BND_SEQUENCE", bnd_fallback );  
    LOG(info) << " bnd_sequence " << bnd_sequence ; 

    std::vector<unsigned> bnd_idx ; 
    qb.getBoundaryIndices( bnd_idx, bnd_sequence, ',' ); 
    LOG(info) << "qb.descBoundaryIndices" << std::endl << qb.descBoundaryIndices( bnd_idx ); 
}


void test_DescDigest(const QBnd& qb )
{
    LOG(info) << std::endl << QBnd::DescDigest(qb.src,8) ; 
}

void test_findName(const QBnd& qb)
{
    std::vector<std::string> names = {
        "Air", 
        "Rock", 
        "Water", 
        "Acrylic",
        "Cream", 
        "vetoWater", 
        "Cheese", 
        "",
        "Galactic", 
        "Pyrex", 
        "PMT_3inch_absorb_logsurf1", 
        "Steel", 
        "Steel_surface",
        "PE_PA",
        "Candy",
        ""
      } ; 

    unsigned i, j ; 

    for(unsigned a=0 ; a < names.size() ; a++ )
    {
         const std::string& n = names[a] ; 
         bool found = qb.findName(i,j,n.c_str() ); 

         std::cout << std::setw(30) << n << " " ; 
         if(found)  
         {
            std::cout 
                << "(" 
                << std::setw(3) << i 
                << ","  
                << std::setw(3) << j
                << ")"
                << " "
                << qb.getItemDigest(i, j )
                ;
         }
         else
         {
            std::cout << "-" ;  
         }
         std::cout << std::endl ;  
    }
}

void test_Add()
{
    const char* cfbase = SOpticksResource::CFBase("CFBASE") ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* optical = NP::Load(cfbase, "CSGFoundry", "optical.npy"); 
    NP* bnd     = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

    LOG(info) << "BEFORE " << std::endl << QBnd::DescOptical(optical, bnd ) << std::endl ; 

    NP* opticalplus = nullptr ; 
    NP* bndplus = nullptr ; 
    std::vector<std::string> specs = { "Rock/perfectAbsorbSurface/perfectAbsorbSurface/Air", "Air///Water" } ;

    QBnd::Add( &opticalplus, &bndplus, optical, bnd, specs ); 

    LOG(info) << "AFTER " << std::endl << QBnd::DescOptical(opticalplus, bndplus ) << std::endl ; 
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

/*
    const char* cfbase = SOpticksResource::CFBase("CFBASE") ; 
    LOG(info) << " cfbase " << cfbase ; 
    NP* bnd = NP::Load(cfbase, "CSGFoundry", "bnd.npy"); 

    QBnd qb(bnd) ; 

    test_descBoundary(qb); 
    test_getBoundaryLine(qb); 
    test_getMaterialLine(qb); 
    test_lookup_technical(qb); 
    test_getBoundarySpec(qb); 
    test_getBoundaryIndices(qb); 

    test_DescDigest(qb); 
    test_findName(qb); 
*/
    test_Add(); 


    return 0 ; 
}
