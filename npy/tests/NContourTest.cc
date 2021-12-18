#include "OPTICKS_LOG.hh"

#include "SSys.hh"
#include "NBBox.hpp"
#include "NNode.hpp"
#include "NNodeSample.hpp"
#include "NContour.hpp"

#include "NCone.hpp"


struct NContourTest
{
    const char* base ; 
    const char* geom ; 
    nnode*      node ; 
    std::vector<float> xx ; 
    std::vector<float> yy ; 

    NContourTest( const char* base, const char* geom ); 
    void scan_save(const char* name); 
};

NContourTest::NContourTest(const char* base_, const char* geom_ )
    :
    base(strdup(base_)),
    geom(strdup(geom_)),
    node(NNodeSample::Sample(geom)) 
{
    nbbox bb = node->bbox() ; 
    NContour::XZ_bbox_grid(xx, yy, bb, 0.01, 0.01 ); 
} 

void NContourTest::scan_save(const char* name)
{
    NContour contour(xx, yy) ; 
    for(unsigned i=0 ; i < xx.size() ; i++ ) 
    for(unsigned j=0 ; j < yy.size() ; j++ ) 
    {
        float sd = (*node)(xx[i], 0.f, yy[j]) ;   // signed distance to nnode surface from grid point 
        contour.setZ( i, j, sd ) ; 
    }
    contour.save(base, geom, name);  
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    const char* geom = SSys::getenvvar("GEOM", "Cone_0") ; 
    LOG(info) << "NContourTest geom [" << geom  << "]"  ; 

    NContourTest t("$TMP/npy/NContourTest", geom);  
    t.scan_save("original"); 

    if(strcmp( geom, "Cone_0") == 0 )
    {
        ncone* cone = (ncone*)t.node ; 

        float _increase_z2 = SSys::getenvfloat("NContourTest_Cone_increase_z2", 0.f ); 
        float _decrease_z1 = SSys::getenvfloat("NContourTest_Cone_decrease_z1", 0.f ); 

        LOG(info) << " _increase_z2 " << _increase_z2 ; 
        LOG(info) << " _decrease_z1 " << _decrease_z1 ; 
     
        if( _increase_z2 != 0.f )  
        {
            cone->increase_z2( _increase_z2 );
        }
        if( _decrease_z1 != 0.f )
        {
            cone->decrease_z1( _decrease_z1 ); 
        }

        t.scan_save("modified"); 
    }
   

    return 0 ; 
} 
