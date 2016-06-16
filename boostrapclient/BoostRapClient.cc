#include <iostream>

#include "BTime.hh"
#include "BDemo.hh"
#include "BCfg.hh"
#include "BLog.hh"
#include "BSys.hh"
#include "BDir.hh"
#include "BJson.hh"



int main(int argc, char** argv)
{

    BLog bl(argc, argv);

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



    std::cerr << " PATH " << BSys::getenvvar("","PATH","no-path-?") << std::endl ; 

 
    typedef std::vector<std::string> VS ; 
    VS names ; 
    BDir::dirlist(names, "C:\\Users\\ntuhep\\env" );
    for(VS::const_iterator it=names.begin() ; it != names.end() ; it++) std::cerr << *it << std::endl ; 


    typedef std::map<std::string, std::string> MSS ; 
    MSS m ; 
    m["hello"] = "world" ;
    m["world"] = "hello" ;
    BJson::saveMap(m, "C:\\Users\\ntuhep", "BoostRapClient.json" );



    return 0 ; 
}
