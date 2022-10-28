// ./SPropTest.sh 

#include "SProp.h"

/**
SPropSet
=========

Note that this is currently booting from the text files. 
That is not a long term solution. 
Will need to boot from an NPFold obtained from SSim 
in production. 

**/

struct SPropSet
{
    SProp* hama ; 
    SProp* nnvt ; 
    SProp* nnvtq ; 
    SProp* pyrex ; 
    SProp* vacuum ; 

    SPropSet(); 
    void getProps(std::vector<SProp*>& props) const ; 
    std::string desc() const ; 
}; 

inline SPropSet::SPropSet()
    :
    hama(new SProp("PMTProperty.R12860", "hama")),
    nnvt(new SProp("PMTProperty.NNVTMCP", "nnvt")),
    nnvtq(new SProp("PMTProperty.NNVTMCP_HiQE", "nnvtq")),
    pyrex(new SProp("Material.Pyrex", "pyrex")),
    vacuum(new SProp("Material.Vacuum", "vacuum"))
{
} 

inline void SPropSet::getProps(std::vector<SProp*>& props) const 
{
    props.push_back(hama); 
    props.push_back(nnvt); 
    props.push_back(nnvtq); 
    props.push_back(pyrex);
    props.push_back(vacuum);
}

inline std::string SPropSet::desc() const
{
    std::vector<SProp*> props ; 
    getProps(props); 

    std::stringstream ss ; 
    for(unsigned i=0 ; i < props.size() ; i++) 
    {
        const SProp* prop = props[i] ; 
        ss << prop->desc() << std::endl ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

void test_meta(const NP* a )
{
    std::cout << " a.sstr " << ( a ? a->sstr() : "-" ) << std::endl ;  
    std::cout << " a.meta [" << a->meta << "]" << std::endl ; 
    std::cout << " a.lpath [" << a->lpath << "]" << std::endl ; 

    a->save("/tmp/t.npy"); 

    std::vector<std::string> lines ; 
    a->get_meta(lines); 
    const std::string& line = lines[0] ; 

    std::cout << " lines.size " << lines.size() << " line [" << line << "]" << std::endl ; 

    std::string other = a->get_meta<std::string>("other", "") ; 

    std::cout << " other[" << other << "]" << std::endl ;
}


int main(int argc, char** argv)
{
    SPropSet ps ; 
    //std::cout << ps.desc() ; 

    NPFold* fold = ps.hama->fold ; 

    const NP* a = fold->get("THICKNESS") ; 
    double d0 = a->get_named_value<double>("ARC_THICKNESS", -1 ); 
    double d1 = a->get_named_value<double>("PHC_THICKNESS", -1 ); 
    std::string units = a->get_meta<std::string>("units", "") ; 

    std::cout << " d0 " << std::scientific << d0 << std::endl ;
    std::cout << " d1 " << std::scientific << d1 << std::endl ;
    std::cout << " units " << units << std::endl;

    return 0 ; 
}
