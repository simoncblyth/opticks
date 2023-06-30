// om- ; TEST=SBndTest om-t

#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "SSim.hh"
#include "SBnd.h"
#include "SPath.hh"
#include "sproplist.h"

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
    sstr::Split( SSys::getenvvar("QCTX_MATERIALS", "Water,LS,Pyrex,Acrylic,NonExisting" ), ',', materials );
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


/**
test_getProperty_Q
-------------------

::

     Q=Air/GROUPVEL SBndTest 
     Q=Air/RINDEX SBndTest 
     Q=Water/RINDEX SBndTest 

**/

void test_getProperty_Q(const SBnd& sb)
{
    const char* q = SSys::getenvvar("Q", "Water/GROUPVEL"); 

    std::vector<std::string> elem ; 
    sstr::Split(q, '/' , elem); 
    assert( elem.size() ==  2); 
    const char* qname = elem[0].c_str(); 
    const char* pname = elem[1].c_str(); 

    std::vector<double> out ; 
    sb.getProperty(out, qname, pname ); 

    std::cout 
        <<  " Q " << q << std::endl 
        << NP::DescSlice(out, 5) 
        ; 
}


void test_getProperty(const SBnd& sb)
{
    const sproplist* pl = sproplist::Material(); 

    std::vector<std::string> pnames ; 
    //SBnd::GetMaterialPropNames(pnames); 
    pl->getNames(pnames) ; 

    for(unsigned i=0 ; i < pnames.size() ; i++) std::cout << pnames[i] << std::endl ; 

    std::vector<std::string> mnames ; 
    sb.getMaterialNames(mnames);     
    //for(unsigned i=0 ; i < mnames.size() ; i++) std::cout << mnames[i] << std::endl ; 

    for(unsigned i=0 ; i < mnames.size() ; i++)
    {
        const char* mname = mnames[i].c_str(); 
        for(unsigned j=0 ; j < pnames.size() ; j++)
        {
            const char* pname = pnames[j].c_str(); 

            std::vector<double> out ; 
            sb.getProperty(out, mname, pname);  

            std::cout 
                <<  " Q " << mname << "/" << pname << std::endl 
                << NP::DescSlice(out, 5) 
                ; 

        }
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);   

    SSim* sim = SSim::Load(); 
    const NP* bnd = sim->get_bnd(); 
    LOG_IF(fatal, !bnd) << " NO bnd : nothing to do " ; 
    if(!bnd) return 0 ; 


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
    test_getPropertyGroup(sb);
    test_DescMaterialProp();
    test_getProperty_Q(sb);
    */
    test_getProperty(sb);

    return 0 ; 
}
// om- ; TEST=SBndTest om-t


