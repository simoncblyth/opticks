
#include "OPTICKS_LOG.hh"
#include "CPho.hh"

void test_basics()
{
    CPho ms ; 
    assert(ms.is_missing()); 
    LOG(info) << " ms " << ms.desc() ; 

    unsigned gs = 42 ; 
    unsigned ix =  0 ; 
    unsigned id = 1000 ; 
    unsigned gn = 0 ; 

    CPho p0(gs, ix, id, gn ); 
    LOG(info) << " p0 " << p0.desc() ; 


    CPho p1 = p0.make_reemit(); 
    
    LOG(info) << " p1 " << p1.desc() ; 

    assert( p0.gs == p1.gs ); 
    assert( p0.ix == p1.ix ); 
    assert( p0.id == p1.id ); 
    assert( p0.gn + 1 == p1.gn ); 
}

void test_generations()
{
    unsigned gs = 42 ; 
    unsigned ix =  0 ; 
    unsigned id = 1000 ; 
    unsigned gn = 0 ; 

    CPho p0(gs, ix, id, gn ); 
    LOG(info) << " p0 " << p0.desc() ; 

    std::vector<CPho> pp ; 
    pp.push_back(p0);   

    for(unsigned i=0 ; i < 10 ; i++)
    {
        CPho p = pp.back().make_reemit() ; 
        pp.push_back(p);
    }


    for(unsigned i=1 ; i < pp.size() ; i++)
    {
        const CPho& parent = pp[i-1]; 
        const CPho& child = pp[i]; 
        LOG(info) << parent.desc() << " " << child.desc() ; 
        assert( parent.id == child.id ); 
        assert( parent.gn + 1 == child.gn ); 
    }
}



int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    test_basics(); 
    test_generations(); 


}


