// om- ; TEST=SBndTest om-t

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SBnd.h"
#include "SPath.hh"
#include "NPFold.h"
    
void test_descBoundary(const SBnd& sb)
{   
    unsigned num_boundary = sb.getNumBoundary();
    LOG(info) 
        << " num_boundary " << num_boundary
        << " sb.descBoundary "
        << std::endl 
        << sb.descBoundary()
        ;
}


void test_getBoundarySpec(const SBnd& sb)
{
    unsigned num_boundary = sb.getNumBoundary();
    LOG(info)
        << " num_boundary " << num_boundary
        ;
    for(unsigned i=0 ; i < num_boundary ; i++)
    {
       std::cout
           << std::setw(4) << i
           << " : "
           << sb.getBoundarySpec(i)
           << std::endl 
           ;
    }
}


void test_getBoundaryLine(SBnd& sb)
{
    const char* spec = SSys::getenvvar("QCTX_SPEC", "Acrylic///LS" );
    unsigned idx = sb.getBoundaryIndex(spec);
    if( idx == SBnd::MISSING )
    {
        LOG(error) << " SBnd MISSING spec " << spec ;
        return ; 
    }

    unsigned num_boundary = sb.getNumBoundary();

    enum { IMAT = 3 } ;
    unsigned line = sb.getBoundaryLine(spec, IMAT);
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
        ;

}


void test_getMaterialLine(SBnd& sb)
{
    std::vector<std::string> materials ;
    SStr::Split( SSys::getenvvar("QCTX_MATERIALS", "Water,LS,Pyrex,Acrylic,NonExisting" ), ',', materials );
    LOG(info) << " materials.size " << materials.size() ;

    for(unsigned i=0 ; i < materials.size() ; i++)
    {
        const char* material = materials[i].c_str() ;
        unsigned line = sb.getMaterialLine(material);
        std::cout
            << " material " << std::setw(50) << material
            << " line " << line
            << std::endl
            ;
    }
}


void test_getBoundaryIndices_0(const SBnd& sb)
{
    const char* bnd_fallback = "Acrylic///LS,Water///Acrylic,Water///Pyrex,Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum" ;
    const char* bnd_sequence = SSys::getenvvar("BND_SEQUENCE", bnd_fallback );
    LOG(info) << " bnd_sequence " << bnd_sequence ;

    std::vector<unsigned> bnd_idx ;
    sb.getBoundaryIndices( bnd_idx, bnd_sequence, ',' );
    LOG(info) << "sb.descBoundaryIndices" << std::endl << sb.descBoundaryIndices( bnd_idx );
}


void test_getBoundaryIndices_1(const SBnd& sb)
{
const char* bnd_fallback = R"LITERAL(
Acrylic///LS
Water///Acrylic
Water///Pyrex
Pyrex/NNVTMCPPMT_PMT_20inch_photocathode_logsurf2/NNVTMCPPMT_PMT_20inch_photocathode_logsurf1/Vacuum
)LITERAL" ; 
    
    const char* bnd_sequence = SSys::getenvvar("BND_SEQUENCE", bnd_fallback );
    LOG(info) << " bnd_sequence " << bnd_sequence ;

    std::vector<unsigned> bnd_idx ;
    sb.getBoundaryIndices( bnd_idx, bnd_sequence, '\n' );
    LOG(info) << "sb.descBoundaryIndices" << std::endl << sb.descBoundaryIndices( bnd_idx );
}



void test_desc(const SBnd& sb )
{
    LOG(info) << std::endl << sb.desc() ; 
}

void test_getMaterialNames( const SBnd& sb )
{
    std::vector<std::string> names ; 
    sb.getMaterialNames(names); 

    LOG(info) << std::endl << SBnd::DescNames(names) ; 
}

void test_getPropertyGroup(const SBnd& sb)
{
    std::vector<std::string> names ; 
    sb.getMaterialNames(names); 

    NPFold* fold = new NPFold ; 
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* material = names[i].c_str(); 
        NP* a = sb.getPropertyGroup( material) ; 
        std::cout << std::setw(20) << material << " " << a->brief() << std::endl; 
        fold->add( material, a ); 
    }
    const char* dir = SPath::Resolve("$TMP/SBndTest/test_getPropertyGroup", DIRPATH); 
    fold->save(dir);
    LOG(info) << dir ;  
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);   

    SSim* sim = SSim::Load(); 
    const NP* bnd = sim->get_bnd(); 

    SBnd sb(bnd) ; 

    /*
    test_descBoundary(sb);
    test_getBoundarySpec(sb);
    test_getBoundaryLine(sb);
    test_getMaterialLine(sb);
    test_getBoundaryIndices_0(sb);
    test_getBoundaryIndices_1(sb);
    test_desc(sb);
    test_getMaterialNames(sb);
    */
    test_getPropertyGroup(sb);

    return 0 ; 
}
// om- ; TEST=SBndTest om-t


