//  nljson-;gcc UseNLJSON.cc -std=c++11 -lstdc++ -I$(nljson-prefix)/include/nljson  -o /tmp/UseNLJSON && /tmp/UseNLJSON

#include <vector>
#include <deque>
#include <list>
#include <map>

#include <string>
#include <iostream>
#include <iomanip>
#include "json.hpp"
using json = nlohmann::json;

namespace ns {
    struct person {
       std::string name ; 
       std::string address ; 
       int age ; 
    };
}

namespace ns {
    void to_json(json& j, const person& p) {
        j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
    }

    void from_json(const json& j, person& p) {
        j.at("name").get_to(p.name);
        j.at("address").get_to(p.address);
        j.at("age").get_to(p.age);
    }
} // namespace ns









void dump(const char* label, const json& j )
{
    std::cout << label << std::endl ; 
 
    std::cout << j << std::endl;
    for (json::const_iterator it = j.begin(); it != j.end(); ++it) std::cout << *it << '\n';

    unsigned nk = j.size() ; 
    std::cout << " size " << nk << std::endl ; 
    std::cout << " is_null " << j.is_null() << std::endl ;   
    std::cout << " is_string " << j.is_string() << std::endl ;   
    std::cout << " is_object " << j.is_object() << std::endl ;   
    std::cout << " is_array " << j.is_array() << std::endl ;   

}


void test_dump()
{
    ns::person p = {"Ned Flanders", "744 Evergreen Terrace", 60};

    json j = p;
    dump("j", j ); 


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

    j2["extra"] =  {1,2,3} ; 

    dump("j2", j2 ); 


    std::vector<int> c_vector {1, 2, 3, 4};
    json j_vec(c_vector);
    // [1, 2, 3, 4]
    dump("j_vec", j_vec); 


    std::deque<double> c_deque {1.2, 2.3, 3.4, 5.6};
    json j_deque(c_deque);
    // [1.2, 2.3, 3.4, 5.6]
    dump("j_deque", j_deque); 

    std::list<bool> c_list {true, true, false, true};
    json j_list(c_list);
    // [true, true, false, true]
    dump("j_list", j_list); 


    std::map<std::string, int> c_map { {"one", 1}, {"two", 2}, {"three", 3} };
    json j_map(c_map);
    // {"one": 1, "three": 3, "two": 2 }

    dump("j_map", j_map); 

    std::string s1 = "Hello, world!";
    json js = s1;
    dump("js", js); 

    json jn ; 
    dump("jn", jn); 
}





void dump_keyed( const char* label, const json& j )
{
    unsigned nk = j.size() ; 
    
    for (json::const_iterator it = j.begin(); it != j.end(); ++it) 
    {
        std::cout 
           << std::setw(10) << it.key()
           << " : "
           << std::setw(10) << it.value()
           << std::endl 
           ;
    }

    json::const_iterator it = j.begin() ; 
    for(unsigned i=0 ; i < nk ; i++)
    {
        std::cout 
            << std::setw(10) << i 
            << " : " 
            << std::setw(10) << it.key()
            << " : " 
            << std::setw(10) << it.value()
            << std::endl 
            ;
        ++it ;   
    }
}



void test_keyed()
{
    json j ;

    j["red"] = "cyan" ; 
    j["green"] = "magenta" ; 
    j["blue"] = "yellow" ; 

    dump_keyed("test_keyed", j ); 
}






void get_kv(const json& j, unsigned i, std::string& k, std::string& v )
{
    json::const_iterator it = j.begin() ; 
    assert( i < j.size() ); 
    std::advance( it, i );
    k = it.key(); 
    v = it.value();  
}

void test_get_kv()
{
    json j ;

    j["red"] = "cyan" ; 
    j["green"] = "magenta" ; 
    j["blue"] = "yellow" ; 

    std::string k, v ; 
    for(unsigned i=0 ; i < j.size() ; i++)
    {
        get_kv(j, i, k, v  ); 

        std::cout 
            << " i:" << std::setw(10) << i
            << " k:" << std::setw(10) << k 
            << " v:" << std::setw(10) << v
            << std::endl
            ; 
    }
}


/**
https://nlohmann.github.io/json/features/arbitrary_types/
https://github.com/nlohmann/json#json-as-first-class-data-type
**/

int main(int argc, char** argv)
{
    std::cout << argv[0] << std::endl ; 

    //test_dump(); 
    //test_keyed(); 
    test_get_kv(); 

    return 0 ; 
}

