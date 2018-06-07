#include <string>
#include <iostream>
#include <fstream>

#include "YGLTF.h"

using ygltf::json ; 


struct sphere
{
    std::string name ; 
    float rmin ; 
    float rmax ; 
};

/*
This autoconv doesnt work, maybe the 
version of the json with ygltf is behind a bit ?


namespace ns {
    // a simple struct to model a person
    struct person {
        std::string name;
        std::string address;
        int age;
    };

    void to_json(json& j, const person& p) {
        j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
    }

    void from_json(const json& j, person& p) {
        p.name = j.at("name").get<std::string>();
        p.address = j.at("address").get<std::string>();
        p.age = j.at("age").get<int>();
    }

} // namespace ns

    ns::person p {"Ned Flanders", "744 Evergreen Terrace", 60};

*/


int main(int argc, char** argv)
{
    // https://github.com/nlohmann/json/

    json j = {} ; 
    
    j["name"] = "hello" ;  
    j["fvalue"] = 1.123 ;
    j["ivalue"] = 123 ;
    j["avalue"] = {1,2,3 } ;

    std::cout << "j\n" <<  j << std::endl ; 

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

    std::cout << "j2\n" << std::setw(4) <<   j2 << std::endl ; 

    return 0 ; 
}
