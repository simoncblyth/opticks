#include <cstring>
#include "OPTICKS_LOG.hh"
#include "SLabelCache.hh"

struct Surf 
{
   const char* name ; 
   int         index ; 
   Surf(const char* name_, int index_) : name(strdup(name_)), index(index_) {}
}; 

struct Turf 
{
   const char* name ; 
   int         index ; 
   Turf(const char* name_, int index_) : name(strdup(name_)), index(index_) {}
}; 


inline std::ostream& operator<<(std::ostream& os, const Surf& s){ os  << "Surf " << s.index << " " << s.name   ; return os; }
inline std::ostream& operator<<(std::ostream& os, const Turf& s){ os  << "Turf " << s.index << " " << s.name   ; return os; }


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info) ; 


    Surf* r = new Surf("red",   100); 
    Surf* g = new Surf("green", 200); 
    Surf* b = new Surf("blue",  300); 

    Turf* c = new Turf("cyan",    1000); 
    Turf* m = new Turf("magenta", 2000); 
    Turf* y = new Turf("yellow",  3000); 
    Turf* k = new Turf("black",   4000); 

    std::vector<const void*> oo = {r,g,b,c,m,y,k} ; 

    // hmm after mixing up the types need to have external info on which is which 
    for(unsigned i=0 ; i < 3         ; i++) std::cout << *(Surf*)oo[i] << std::endl ; 
    for(unsigned i=3 ; i < oo.size() ; i++) std::cout << *(Turf*)oo[i] << std::endl ; 


    SLabelCache<int> cache(-1); 

    for(unsigned i=0 ; i < oo.size() ; i++)
    {
        const void* o = oo[i] ; 
        cache.add(o, int(i)); 
        int idx = cache.find(o); 
        assert( idx == int(i) ); 
    }

    const void* anon = (const void*)m ; 
    int idx_m = cache.find(anon) ;  
    assert( idx_m == 4 ); 

    return 0 ; 
}


