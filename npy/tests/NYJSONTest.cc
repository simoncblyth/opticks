/*


JSON for Modern C++ : a C++11 header-only JSON class

* https://github.com/nlohmann/json
* https://nlohmann.github.io/json/
* http://www.oss.io/p/nlohmann/json

*/


#include <exception>
#include <fstream>

#include "NYJSON.hpp"
#include "PLOG.hh"

using json = nlohmann::json;


struct json_exception : std::exception {
    /// constructor with error message
    json_exception(const std::string& errmsg) : _errmsg(errmsg) {}

    /// retieval of error message
    virtual const char* what() const throw() { return _errmsg.c_str(); }

   private:
    std::string _errmsg;
};



void test_iterate(const json& j)
{
    // iterate the array
    LOG(info) << " values " ;
    for (json::const_iterator it = j.begin(); it != j.end(); ++it) {
      std::cout << *it << '\n';
    }

    LOG(info) << " key : values " ;
    for (json::const_iterator it = j.begin(); it != j.end(); ++it) {
      std::cout << it.key() << " : " << it.value() << "\n";
    }
}


void test_build()
{
    // create an empty structure (null)
    json j;

    // add a number that is stored as double (note the implicit conversion of j to an object)
    j["pi"] = 3.141;

    // add a Boolean that is stored as bool
    j["happy"] = true;

    // add a string that is stored as std::string
    j["name"] = "Niels";

    // add another null object by passing nullptr
    j["nothing"] = nullptr;

    // add an object inside the object
    j["answer"]["everything"] = 42;

    // add an array that is stored as std::vector (using an initializer list)
    j["list"] = { 1, 0, 2 };

    // add another object (using an initializer list of pairs)
    j["object"] = { {"currency", "USD"}, {"value", 42.99} };

    LOG(info) << j.dump(4) ;

    test_iterate(j);


    json jcopy(j);
 
    LOG(info) << "jcopy: " << jcopy.dump(4) ;

}


void test_build2()
{
    json j2 = {
      {"pi", 3.141},
      {"happy", true},
      {"name", "Niels"},
      {"nothing", nullptr},
      {"answer", {
        {"everything", 42}
      }},
      {"list", {1, 0, 2}},
      {"object", {
        {"currency", "USD"},
        {"value", 42.99}
      }}
    };

    LOG(info) << j2.dump(4) ;
}

void test_deserialize()
{
    // create object from string literal
    json j = "{ \"happy\": true, \"pi\": 3.141 }"_json;
    LOG(info) << j.dump(4) ;
    LOG(info) << j.dump() ;

    // or even nicer (thanks http://isocpp.org/blog/2015/01/json-for-modern-cpp)
    auto j2 = R"(
      {
        "happy": true,
        "pi": 3.141
      }
    )"_json;

    LOG(info) << j2.dump(4) ;

    // or explicitly
    auto j3 = json::parse("{ \"happy\": true, \"pi\": 3.141 }");


    LOG(info) << j3.dump(4) ;

    std::cout << j3 << std::endl ;  // doesnt work with LOG 
    std::cout << std::setw(4) << j3 << std::endl ;  
}

void test_json_from_map()
{
    std::map<std::string, int> c_map { {"one", 1}, {"two", 2}, {"three", 3} };
    json j(c_map);

    LOG(info) << j.dump(4) ;
}





void test_load( json& js , const char* path )
{
    try {
        std::ifstream stream(path);
        if (!stream) throw json_exception("test_load : could not open stream ");
        stream >> js;
    } catch (const std::exception&) {
        throw json_exception("test_load : could not load json");
    }    
}

void test_load( const char* path )
{
    json js ; 
    test_load( js, path );
    
    LOG(info)
          << " path " << path 
          << " js " << js.dump(4)
          ;    
}

bool test_save( const json& js, const char* path)
{
    auto f = fopen(path, "wt");
    if (!f) {
        return false;
    }    

    std::string txt = js.dump(4) ; 
    fwrite( txt.c_str(), 1, (int)txt.size(), f);
    fclose(f);

    return true ; 
}


void test_get_explicit()
{
    json j2 = {
      {"pi", 3.141},
      {"happy", true},
      {"name", "Niels"},
      {"nothing", nullptr},
      {"answer", {
        {"everything", 42}
      }},
      {"list", {1, 0, 2}},
      {"object", {
        {"currency", "USD"},
        {"value", 42.99}
      }}
    };


    std::string name = j2["name"].get<std::string>() ; 
    float fpi = j2["pi"].get<float>() ;
    int ipi = j2["pi"].get<int>() ;
    unsigned uans = j2["answer"]["everything"].get<unsigned>(); 

    LOG(info) 
        << " name " << name  
        << " fpi " << fpi  
        << " ipi " << ipi  
        << " uans " << uans
        ; 


}




int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    test_build();
    /*
    test_build2();
    test_deserialize();
    test_json_from_map();
    */

    
    test_get_explicit();


    return 0 ; 
}

