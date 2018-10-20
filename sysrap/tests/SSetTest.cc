// TEST=SSetTest om-t

#include <cassert>
#include <map>
#include <set>

#include "OPTICKS_LOG.hh"

void test_set()
{
    std::set<unsigned> a ; 

    a.insert(1) ; 
    a.insert(2) ; 
    a.insert(2) ; 
    a.insert(3) ; 

    assert( a.size() == 3 ); 
}


void dump( const std::map<unsigned, std::set<unsigned> >& ms, unsigned i )
{
    typedef std::set<unsigned> SU ; 
    const SU& s = ms.at(i) ; 
    std::cout 
          << " s.size() " << s.size() 
          << " ( " 
          ; 

    for(SU::const_iterator it=s.begin() ; it != s.end() ; it++ ) 
          std::cout << *it << " " ; 
 
    std::cout 
          << " ) " 
          << std::endl 
          ;

} 

void test_mapset()
{
    std::map<unsigned, std::set<unsigned> > a ; 

    a[0].insert(1) ; 
    a[0].insert(2) ; 
    a[0].insert(2) ; 
    a[0].insert(3) ; 

    dump(a, 0); 

    a[1].insert(1) ; 
    a[1].insert(2) ; 
    a[1].insert(2) ; 
    a[1].insert(3) ; 

    dump(a, 1); 

    assert( a[0].size() == 3 ); 
    assert( a[1].size() == 3 ); 

}





int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    test_set(); 
    test_mapset();  

   
    return 0 ; 

}

