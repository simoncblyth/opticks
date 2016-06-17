#include <iostream>

#include "BTime.hh"
#include "BDemo.hh"
#include "BCfg.hh"
#include "BLog.hh"
#include "BSys.hh"
#include "BDir.hh"

//#include "BJson.hh"
#include "BMap.hh"
#include "BList.hh"

#include "BStr.hh"
#include "BHex.hh"
#include "BFile.hh"

#include "BDigest.hh"
#include "BArrayDigest.hh"




int main(int argc, char** argv)
{

    BLog blog(argc, argv);

    LOG(info)  
              << " argc " << argc 
              << " argv[0] " << argv[0] 
              ;

    std::cerr 
              << " argc " << argc 
              << " argv[0] " << argv[0] 
              << std::endl ; 

    BTime bt ; 
    std::cerr << bt.check() << std::endl ; 
    std::cout << BTime::now("%Y",0) << std::endl ; 

    BDemo bd(42);
    bd.check();

    std::cerr << "checked" << std::endl ; 

    BCfg bc("bcfg", false);
    std::cerr << bc.getName() << std::endl ; 


    std::string path = BFile::FormPath("~", "BoostRapClient.txt" );
    std::cerr << " path " << path << std::endl ; 

    std::string home = BFile::FormPath("~");
    std::cerr << " home " << home << std::endl ; 



    std::cerr << " PATH " << BSys::getenvvar("","PATH","no-path-?") << std::endl ; 
    const char* dir = home.c_str() ;

 
    typedef std::vector<std::string> VS ; 
    VS names ; 
    BDir::dirlist(names, dir );
    for(VS::const_iterator it=names.begin() ; it != names.end() ; it++) std::cerr << *it << std::endl ; 


    typedef std::map<std::string, std::string> MSS ; 
    MSS m ; 
    m["hello"] = "world" ;
    m["world"] = "hello" ;


    const char* mapname = "BoostRapClient.BMap.json" ;
    const char* lisname = "BoostRapClient.BList.json" ;

    BMap<std::string,std::string> bm(&m); 
    bm.save(dir,  mapname );

    MSS m2 ; 
    BMap<std::string,std::string> bm2(&m2); 
    bm2.load(dir, mapname );
    bm2.dump("bm2");


    typedef std::pair<std::string, std::string>  PSS ;
    typedef std::vector<PSS> VSS ;
    VSS l ; 
    l.push_back(PSS("a","red"));
    l.push_back(PSS("b","green"));
    l.push_back(PSS("c","blue"));


    BList<std::string,std::string> bl(&l); 
    bl.save(dir, lisname);

    VSS l2 ;  
    BList<std::string,std::string> bl2(&l2); 
    bl2.load(dir, lisname);
    bl2.dump("bl2");


    assert(BStr::listHasKey("red,green,blue", "green", ",") == true); 
    assert(BStr::listHasKey("red,green,blue", "cyan", ",") == false ); 

    BHex<int> bh(15) ;
    std::cout << " as_hex " << bh.as_hex() << std::endl ; 


    const char* buf = "hello" ;
    std::string digest = BDigest::md5digest(buf, strlen(buf));
    std::cout << " digest of " << buf << " is " << digest << std::endl ; 


    int arr[3]; 
    arr[0] = 0 ;  
    arr[1] = 1 ;  
    arr[2] = 2 ;

    std::string adigest = BArrayDigest<int>::arraydigest(arr, 3);
    std::cout << " adigest  " << adigest << std::endl ; 


  


    //BJson::saveMap(m, "C:\\Users\\ntuhep", "BoostRapClient.json" );

    return 0 ; 
}
