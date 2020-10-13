#include <vector>
#include <string>
#include <limits>

#include "OPTICKS_LOG.hh"
#include "BOpticksResource.hh"
#include "BFile.hh"
#include "NPY.hpp"
#include "NGLM.hpp"
#include "numpy.hpp"


struct numpyTest
{
    std::vector<int> shape ;
    std::vector<unsigned> data ;


    glm::vec4 getQuadLocal(NPY<unsigned>* a, unsigned i, unsigned j, unsigned k)
    {
        glm::vec4 vec ;
        for(unsigned int l=0 ; l < 4 ; l++) vec[l] = a->getValue(i,j,k,l);
        //if( i >= 3199 && i <= 3199+10 ) std::cout << "(" << i << ")" << glm::to_string(vec) << std::endl ;  
        return vec ; 
    }

    numpyTest( const char* path, int mode )
    {
        std::cout << path << std::endl ; 

        if( mode == 0)
        {
            LOG(info) << " aoba::LoadArrayFromNumpy " ; 
            aoba::LoadArrayFromNumpy<unsigned>(path, shape, data );
        }
        else if( mode == 1)
        {
            LOG(info) << " NPY<unsigned>::load, getValuesConst " ; 
            NPY<unsigned>* a = NPY<unsigned>::load(path) ; 

            const std::vector<int>& sh = a->getShapeVector(); 
            for(unsigned d=0 ; d < sh.size() ; d++)  shape.push_back(sh[d]) ; 

            unsigned nv = a->getNumValues(); 
            const unsigned* values = a->getValuesConst(); 
            for(unsigned v=0 ; v < nv ; v++) data.push_back(values[v]) ; 
        }
        else if( mode > 1)
        {
            if( mode == 2 ) LOG(info) << " NPY<unsigned>::load, getValue " ; 
            if( mode == 3 ) LOG(info) << " NPY<unsigned>::load, getQuadF " ; 
            if( mode == 4 ) LOG(info) << " NPY<unsigned>::load, getQuad_ " ; 
            if( mode == 5 ) LOG(info) << " NPY<unsigned>::load, getQuadU " ; 
            if( mode == 6 ) LOG(info) << " NPY<unsigned>::load, getQuadI " ; 
            if( mode == 7 ) LOG(info) << " NPY<unsigned>::load, getQuadLocal " ; 

            NPY<unsigned>* a = NPY<unsigned>::load(path) ; 

            const std::vector<int>& sh = a->getShapeVector(); 
            for(unsigned d=0 ; d < sh.size() ; d++)  shape.push_back(sh[d]) ; 

            assert( sh.size() == 2 && sh[1] == 4 ); 
            unsigned ni = sh[0] ; 
            unsigned nj = sh[1] ; 
            for(unsigned i=0 ; i < ni ; i++)
            {
                if( mode == 2 )
                {
                    for(unsigned j=0 ; j < nj ; j++) 
                    {
                        unsigned v = a->getValue(i,j,0);  
                        data.push_back(v); 
                    } 
                }
                else if( mode > 2)
                {                              
                    glm::uvec4 q ; 
                    if(mode == 3 ) q = a->getQuadF(i); 
                    if(mode == 4 ) q = a->getQuad_(i); 
                    if(mode == 5 ) q = a->getQuadU(i); 
                    if(mode == 6 ) q = a->getQuadI(i); 
                    if(mode == 7 ) q = getQuadLocal(a, i, 0, 0); 

                    data.push_back(q.x);  
                    data.push_back(q.y);  
                    data.push_back(q.z);  
                    data.push_back(q.w);  
                }
            }
        }
 

    }

    void dump()
    {
        std::cout << " shape ( " ;
        for(unsigned i=0 ; i < shape.size() ; i++) std::cout << shape[i] << " " ; 
        std::cout << " ) " << std::endl ; 
    }

    void dump(unsigned i0, unsigned i1)
    {
        dump(); 
        unsigned num_dim = shape.size() ;
        if( num_dim != 2 ) return ; 
        unsigned ni = shape[0] ; 
        unsigned nj = shape[1] ; 
        for(unsigned i=std::min(i0,ni) ; i < std::min(i1,ni) ; i++)
        {
            std::cout << "(" << std::setw(5) << std::dec << i << ") " ; 
            for(unsigned j=0 ; j < nj ; j++)
            {
                 unsigned index = i*nj + j ; 
                 if( j == 0 )
                     std::cout << std::setw(10) << std::dec << data[index]  ;  
                 else 
                     std::cout << std::setw(10) << std::hex << data[index] << std::dec ;  
                 
            }
            std::cout << std::endl ; 
        }
    }

}; 


template<typename T>
std::string desc(const char* label, T val)
{
    std::stringstream ss ; 
    ss 
       << label 
       << " ("
       << std::hex << val 
       << " "
       << std::dec << val 
       << ")"
       ;
    return ss.str(); 
}

/**
test_unsigned_float
--------------------

Going through float corrupts unsigned beyond 0x1 << 24, 16.777216M::

    In [10]: np.uint32(np.float32(np.uint32(0+(0x1 << 24)))) == 0+(0x1 << 24)
    Out[10]: True

    In [11]: np.uint32(np.float32(np.uint32(1+(0x1 << 24)))) == 1+(0x1 << 24)
    Out[11]: False

    In [14]: "{0:x} {0}".format(0x1 << 24)
    Out[14]: '1000000 16777216'



**/

void test_unsigned_float()
{
    unsigned count = 0 ; 
    unsigned mx = std::numeric_limits<unsigned>::max() ;

    for(unsigned i=0 ; i < mx ; i++)
    {
        unsigned i0 = i ;
        float    f0 = i0 ; 
        unsigned i1 = unsigned(f0);   
        if( i1 != i0 ) 
        {
            count += 1 ; 
            std::cout 
                << desc<unsigned>(" i0",i0)
                << desc<unsigned>(" i1",i1)
                << std::endl
                ; 
            if(count > 20) break ; 
        }
    }
    LOG(info) 
       << desc<unsigned>("mx",mx) 
       << desc<unsigned>("cn",count) 
       ;
}  


void test_getters(const char* path)
{
    LOG(info); 

    if( BFile::ExistsFile(path) == false )
    {
        LOG(fatal) << " path does not exist " << path  ; 
        return ; 
    } 

    unsigned i0 = 3199 ; 
    unsigned i1 = i0+10 ; 
    unsigned num_mode = 8 ; 

    for( unsigned mode=0 ; mode < num_mode ; mode++)
    {
        numpyTest* nt = new numpyTest(path, mode); 
        nt->dump(i0,i1); 
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* default_path = BOpticksResource::GetCachePath("GNodeLib/all_volume_identity.npy");
    const char* path = argc > 1 ? argv[1] : default_path ;  

    //test_unsigned_float(); 
    test_getters(path); 

    return 0 ; 
}
// om-;TEST=numpyTest om-t
