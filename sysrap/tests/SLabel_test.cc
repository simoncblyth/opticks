// ./SLabel_test.sh
#include <iostream>
#include <iomanip>
#include <cstring>
#include <string>

#include "SLabel.h"

void test_GEOMLoad()
{
    SLabel* id = SLabel::GEOMLoad(); 
    std::cout 
        << "test_GEOMLoad"
        << std::endl 
        << "SLabel* id = SLabel::GEOMLoad() ; id->detail() " 
        << std::endl 
        << id->detail()
        << std::endl 
        ; 
}


void test_findIndicesWithListedLabels_one_by_one()
{
    SLabel* id = SLabel::GEOMLoad(); 
    for(unsigned i=0 ; i < id->label.size() ; i++)
    {
        std::vector<unsigned> indices ; 
        id->findIndicesWithListedLabels(indices, id->label[i].c_str(), ',' ); 
        assert( indices.size() == 1 && indices[0] == i );  
    }
}

void test_findIndicesWithListedLabels_all()
{
    SLabel* id = SLabel::GEOMLoad(); 

    char delim = ',' ; 

    unsigned num_label = id->label.size() ; 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_label ; i++)
    {
        ss << id->label[i].c_str() << ( i < num_label - 1 ? delim : '\0' ) ; 
    }

    std::string _ls = ss.str(); 
    const char* ls = _ls.c_str(); 

    std::cout << "[" << ls << "]\n" ;  

    std::vector<unsigned> indices ; 
    id->findIndicesWithListedLabels(indices, ls, delim ); 
    assert( indices.size() == num_label );  
}

void test_IsIdxLabelListed()
{
    SLabel* id = SLabel::GEOMLoad(); 

    char delim = ',' ; 
    std::vector<std::string> labs ; 
    unsigned num_label = id->label.size() ; 
    for(unsigned i=0 ; i < num_label ; i++)
    {
        if(i % 2 == 0) labs.push_back( id->label[i].c_str() ); 
    }

    unsigned num_labs = labs.size(); 
    std::stringstream ss ; 
    for(unsigned i=0 ; i < num_labs ; i++)
    {
        ss << labs[i].c_str() << ( i < num_labs - 1 ? delim : '\0' ) ;     
    } 
    std::string _ls = ss.str(); 
    const char* ls = _ls.c_str(); 
    std::cout << ls << "\n" ; 

    for(unsigned i=0 ; i < num_label ; i++)
    {
        bool listed = SLabel::IsIdxLabelListed(id->label, i, ls, ',' ); 
        std::cout << " i " << i << " listed " << ( listed ? "Y" : "N" ) << " " << id->label[i] << "\n" ;  
    }
}





int main()
{
    //test_GEOMLoad(); 
    //test_findIndicesWithListedLabels_one_by_one(); 
    //test_findIndicesWithListedLabels_all(); 

    test_IsIdxLabelListed();


    return 0 ; 
}
