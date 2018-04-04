#include "BStr.hh"
#include <iomanip>

#include <iterator>
#include <iostream>

#include <cstdio>
#include <boost/algorithm/string/replace.hpp>

#include "BRAP_LOG.hh"
#include "PLOG.hh"


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



int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

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
*/

    test_Split();

    return 0 ; 
}

