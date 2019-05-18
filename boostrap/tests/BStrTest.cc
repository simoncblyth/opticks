// TEST=BStrTest om-t 
#include "BStr.hh"
#include <iomanip>

#include <iterator>
#include <iostream>

#include <cstdio>
#include <boost/algorithm/string/replace.hpp>

#include "OPTICKS_LOG.hh"


void test_index_all()
{
    LOG(info) << "." ; 

    std::vector<std::string> v ; 
    v.push_back("red");
    v.push_back("green");
    v.push_back("red");
    v.push_back("blue");

    std::vector<unsigned> ii ; 
    ii.clear();
    assert( BStr::index_all(ii, v, "red") == 2 );
    assert( ii.size() == 2 );

    ii.clear(); 
    assert( BStr::index_all(ii, v, "green") == 1 );
    assert( ii.size() == 1 );

    ii.clear(); 
    assert( BStr::index_all(ii, v, "blue") == 1 );
    assert( ii.size() == 1 );

    ii.clear(); 
    assert( BStr::index_all(ii, v, "cyan") == 0 );
    assert( ii.size() == 0 );
}



void test_index_first()
{
    LOG(info) << "." ; 

    std::vector<std::string> v ; 
    v.push_back("red");
    v.push_back("green");
    v.push_back("blue");

    assert( BStr::index_first(v, "red") == 0 );
    assert( BStr::index_first(v, "green") == 1 );
    assert( BStr::index_first(v, "blue") == 2 );
    assert( BStr::index_first(v, "cyan") == -1 );
}






void test_ijoin()
{
    std::vector<int> elem ; 
    std::string ij = BStr::ijoin(elem, ',');
    assert( strcmp(ij.c_str(), "") == 0 );
}



void test_fsplit()
{
    const char* line = "1.12,10.0,-100.1,-200,+20.5" ;

    std::vector<float> elem ; 
    BStr::fsplit(elem, line, ',');

    LOG(info) << " fsplit [" << line << "] into elem count " << elem.size() ; 
    for(unsigned i=0 ; i < elem.size() ; i++) std::cout << elem[i] << std::endl ; 

    assert( elem.size() == 5 );
    assert( elem[0] == 1.12f );
    assert( elem[1] == 10.f );
    assert( elem[2] == -100.1f );
    assert( elem[3] == -200.f );
    assert( elem[4] == 20.5f );

}


void test_isplit()
{
    const char* line = "1,10,100,-200" ;

    std::vector<int> elem ; 
    BStr::isplit(elem, line, ',');

    LOG(info) << " isplit [" << line << "] into elem count " << elem.size() ; 

    assert(elem.size() == 4);
    assert(elem[0] == 1 );
    assert(elem[1] == 10 );
    assert(elem[2] == 100 );
    assert(elem[3] == -200 );

    std::string ij = BStr::ijoin(elem, ',');

    LOG(info) << " ijoin elem into [" << ij << "]" ;

    assert( strcmp( ij.c_str(), line) == 0);
}



void test_usplit()
{
    const char* s = "0:5,7,10:15,101,200:210" ; 

    std::vector<unsigned> elem ; 
    BStr::usplit(elem, s, ',') ; 

    LOG(info) 
        << " s " << s
        << " elem.size() " << elem.size()  
        ;

    for(unsigned i=0 ; i < elem.size() ; i++ ) std::cout << elem[i] << std::endl ;  
}


void test_pair_split()
{
     const char* s = "0:0,1:10,2:20,3:30,4:40,5:-50" ; 

     typedef std::pair<int,int> II ; 
     typedef std::vector<II> VII ; 

     VII vii ; 
     BStr::pair_split( vii, s, ',', ":" ); 
     assert( vii.size() == 6 );

    
     for(unsigned i=0 ; i < vii.size() ; i++)
     {
         LOG(info) 
               << "( " 
               << std::setw(3) << i
               << ") "   
               << std::setw(10) << vii[i].first 
               << " : " 
               << std::setw(10) << vii[i].second 
               ;
     }


}





void test_afterLastOrAll_(const char* s, const char* x )
{
    char* a = BStr::afterLastOrAll(s) ; 
    bool match = strcmp(a,x) == 0 ; 

    if(!match)
    {
        LOG(fatal) << " MISMATCH " 
                   << " s " << std::setw(30) << s 
                   << " x " << std::setw(30) << x 
                   << " a " << std::setw(30) << a
                   ;
    }

    assert(match); 
}


void test_afterLastOrAll()
{
    test_afterLastOrAll_("/hello/dear/world/take/me","me") ; 
    test_afterLastOrAll_("me","me") ; 
    test_afterLastOrAll_("me/","me/") ; 
}

void test_DAEIdToG4_(const char* daeid, const char* x_g4name, bool trimPtr)
{
    char* g4name = BStr::DAEIdToG4(daeid, trimPtr);
    bool match = strcmp( g4name, x_g4name  ) == 0 ;

    if(!match) 
    { 
    LOG(debug) 
                  << " " << ( match ? "match" : "MISMATCH" )
                  << " daeid " << daeid 
                  << " g4name " << g4name 
                  << " x_g4name " << x_g4name 
                  ;
    }

    assert(match);
}

void test_DAEIdToG4()
{
    test_DAEIdToG4_("__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40", "/dd/Geometry/PoolDetails/lvLegInIWSTub", true );
    test_DAEIdToG4_("__dd__Geometry__PoolDetails__lvLegInIWSTub0xc400e40", "/dd/Geometry/PoolDetails/lvLegInIWSTub0xc400e40", false );

    test_DAEIdToG4_(
    "__dd__Geometry__Sites__lvNearHallBot--pvNearPoolDead0xc13c018",
    "/dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c018",
    false);

}



void test_patternPickField()
{
    std::string str = "aaaa__bbbb__cccccccccccccc__d__e" ;
    std::string ptn = "__" ;

    for(int field=-5 ; field < 5 ; field++ )
    {
        printf("patternPickField(%s,%s,%d) --> ", str.c_str(), ptn.c_str(), field  );
        std::string pick = BStr::patternPickField(str, ptn,field);
        printf(" %s \n", pick.c_str());
    }
}

void test_StartsWith()
{
    std::string s = "hello_cruel" ; 
    std::string q = "hell" ; 

    assert( BStr::StartsWith(s.c_str(), "hell") );
    assert( !BStr::StartsWith(s.c_str(), " hell") );

}


void test_ReplaceAll()
{

    const char* r = R"glsl(

    uniform MatrixBlock  
    {   
        mat4 ModelViewProjection;
    } ; 

)glsl";

    const char* s = R"glsl(

    #version 400 core

    $UniformBlock

    layout (location = 0) in vec4 VertexPosition;
    layout (location = 1) in mat4 VizInstanceTransform ;

    void main()
    {   
        gl_Position = ModelViewProjection * VizInstanceTransform * VertexPosition ;
    }   

)glsl";


    std::string tmpl = s ;
    BStr::ReplaceAll(tmpl, "$UniformBlock", r );

    std::cout << std::endl << tmpl <<  std::endl ;

 
}

void test_ekv_split()
{
   std::vector<std::pair<std::string, std::string> > ekv ; 

   //const char* line_ = "TO:0 SC: SR:1 SA:0" ; 
   const char* line_ = "TO:0 SC SR:1 SA:0" ; 

   char edelim = ' ' ; 
   const char* kvdelim=":" ; 

   BStr::ekv_split( ekv, line_, edelim, kvdelim );

   LOG(info) << line_ ; 

   for(unsigned i=0 ; i < ekv.size() ; i++ ) 
         std::cout << "[" 
                   << ekv[i].first 
                   << "] -> [" 
                   <<  ekv[i].second << "]" 
                   << ( ekv[i].second.empty() ? "EMPTY" : "" )
                   << std::endl ; 


}

void test_replace_all()
{
    std::string s = "--c2max_0.5" ; 
    BStr::replace_all(s, "_", " ");

    LOG(info) << s  ;

}

void test_LexicalCast()
{
    bool badcast(false);

    unsigned u = BStr::LexicalCast<unsigned>( "101", -1 , badcast );
    assert( u == 101 && !badcast );

    int i  = BStr::LexicalCast<int>( "-101", -1 , badcast );
    assert( i == -101 && !badcast );

    float f  = BStr::LexicalCast<float>( "-101.1", -1 , badcast );
    assert( f == -101.1f && !badcast );
}


template <typename T>
void test_Split(const char* s, unsigned xn)
{
    std::vector<T> v ; 
    unsigned n = BStr::Split<T>(v, s, ',' );

    LOG(info) 
        << " n:" << n 
        << " xn:" << xn 
        << " v:" << v.size()
        ;

    assert( n == xn && n == v.size() ) ; 
    for(unsigned i=0 ; i < n ; i++ ) std::cout << " " << v[i] ; 
    std::cout << std::endl ; 

}

void test_Split()
{
    test_Split<unsigned>( "0,1,2,3,4,5,6,7,8,9", 10 );
    test_Split<int>( "0,1,2,-3,4,5,6,-7,8,9", 10 );
    test_Split<float>("0.5,1.5,2,3,4,5,6,7,8,9", 10 );
}

void test_Split1(const char* keys, unsigned expected_nkey )
{
    std::vector<std::string> elem ;
    BStr::split(elem, keys, ',') ;

    unsigned nkey = elem.size() ;

    bool expected = nkey == expected_nkey ;

    if(!expected)
    {
         LOG(fatal) << " keys " << keys 
                    << " nkey " << nkey 
                    << " expected_nkey " << expected_nkey
                    ; 
    }

    assert( expected );
}

void test_utoa()
{
    LOG(info) << "." ; 
    int width = 3 ; 
    bool zeropad = true ; 
    for(unsigned i=0 ; i < 2000 ; i+= 100 )
    {
        std::cout << i << " : " << BStr::utoa(i, width, zeropad ) << std::endl ; 
    }
}

void test_Contains()
{
   assert( BStr::Contains("/some/path/to/VolCathodeEsque", "Cathode,cathode", ',' ) == true ) ; 
   assert( BStr::Contains("/some/path/to/VolcathodeEsque", "Cathode,cathode", ',' ) == true ) ; 
   assert( BStr::Contains("/some/path/to/Nowhere", "Cathode,cathode", ',' ) == false ) ; 
}

void test_WithoutEnding()
{
    const char* s = "lvPmtHemiCathodeSensorSurface" ; 
    const char* q = "SensorSurface" ; 
    assert( BStr::EndsWith(s, q) ); 
    const char* a = BStr::WithoutEnding(s, q) ; 
    const char* x = "lvPmtHemiCathode" ; 
    assert( strcmp( a, x ) == 0); 
}

void test_GetField()
{
    const char* name = "BS:007:__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" ; 

    const char* f0 = "BS" ; 
    const char* f1 = "007" ; 
    const char* f2 = "__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" ; 

    assert( BStr::NumField(name, ':') == 3 ); 

    assert( strcmp( BStr::GetField( name, ':', 0 ), f0 ) == 0 ); 
    assert( strcmp( BStr::GetField( name, ':', 1 ), f1 ) == 0 ); 
    assert( strcmp( BStr::GetField( name, ':', 2 ), f2 ) == 0 ); 
    assert( strcmp( BStr::GetField( name, ':', -1 ), f2 ) == 0 ); 
    assert( strcmp( BStr::GetField( name, ':', -2 ), f1 ) == 0 ); 
    assert( strcmp( BStr::GetField( name, ':', -3 ), f0 ) == 0 ); 

}


void test_ctoi()
{
    char c = '4' ; 
    int i = (int)c - (int)'0' ; 
    assert( i == 4 ) ; 
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

/*
    test_patternPickField();
    test_afterLastOrAll();
    test_isplit();
    test_ijoin();
    test_fsplit();
    test_StartsWith();
    test_ReplaceAll();

    test_index_first();
    test_index_all();
    test_DAEIdToG4();  
    test_ekv_split();
    test_replace_all();
    test_usplit();
    test_LexicalCast();
    test_Split();

    test_Split1( "RINDEX,ABSLENGTH,RAYLEIGH,REEMISSIONPROB", 4 );
    test_Split1( "RINDEX,,,", 3 );
    test_Split1( "RINDEX,,, ", 4 );
    test_utoa();
    test_Contains();
    test_WithoutEnding();
    test_GetField();
    test_ctoi();
*/
    test_pair_split();


    return 0 ; 
}

