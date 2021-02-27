// name=GridSpecTest.cc ; gcc $name -std=c++11 -lstdc++  -o /tmp/$name && /tmp/$name
#include <array>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cassert>

void parse_gridspec(std::array<int,9>& grid, const char* spec)
{
    int idx = 0 ; 
    std::stringstream ss(spec); 
    std::string s;
    while (std::getline(ss, s, ',')) 
    {
        std::stringstream tt(s); 
        std::string t;
        while (std::getline(tt, t, ':')) grid[idx++] = std::atoi(t.c_str()) ; 
    }
    for(int i=0 ; i < 9 ; i++) std::cout << grid[i] << " " ; 
    std::cout << std::endl ;           
}

int main(int argc, char** argv)
{
    const char* spec = "-10:11:2,-10:11:2,-10:11:2" ; 
    std::array<int,9> grid ; 
    parse_gridspec(grid, spec); 

    for(int i=grid[0] ; i < grid[1] ; i+=grid[2] )
    for(int j=grid[3] ; j < grid[4] ; j+=grid[5] )
    for(int k=grid[6] ; k < grid[7] ; k+=grid[8] )
    {
        std::cout << "(" << i << "," << j << "," << k << ")" << std::endl ; 
    }


    return 0 ; 
}
