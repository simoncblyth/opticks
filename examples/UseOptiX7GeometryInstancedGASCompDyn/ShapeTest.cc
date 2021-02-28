// name=ShapeTest ; mkdir -p /tmp/ShapeTestWrite/{0,1} ; gcc -g $name.cc Shape.cc Sys.cc -lstdc++ -std=c++11 -I. -o /tmp/$name && lldb_ /tmp/$name

#include <cassert>
#include <iostream>
#include <vector>
#include "Shape.h"

void test_create_write_0()
{
    Shape s0('S', 100.f); 
    std::cout << s0.desc() << std::endl ;  
    s0.write("/tmp", "ShapeTestWrite", 0 ); 
}

void test_create_write_1()
{
    std::vector<float> szs = { 100.f, 50.f } ; 
    Shape s1("SS", szs ); 
    std::cout << s1.desc() << std::endl ;  
    s1.write("/tmp", "ShapeTestWrite", 1 ) ;
}


// gives ShapeTest(84752,0x7fff9a943380) malloc: *** error for object 0x100500000: pointer being freed was not allocated
void test_collect_obj(std::vector<Shape>& shapes)
{
    Shape s0('S', 100.f ); 
    Shape s1('S', 100.f ); 
    shapes.push_back(s0); 
    shapes.push_back(s1); 
}
void test_collect_obj()
{
    std::vector<Shape> shapes ; 
    test_collect_obj(shapes); 
    assert(shapes.size() == 2); 
}

void test_collect_ptr(std::vector<Shape*>& shapes)
{
    Shape* s0 = new Shape('S', 100.f ); 
    Shape* s1 = new Shape('S', 100.f ); 
    shapes.push_back(s0); 
    shapes.push_back(s1); 
}
void test_collect_ptr()
{
    std::vector<Shape*> shapes ; 
    test_collect_ptr(shapes); 
    assert(shapes.size() == 2); 
}



int main(int argc, char** argv)
{
    test_create_write_0(); 
    test_create_write_1(); 
    //test_collect_obj(); 
    test_collect_ptr(); 
    return 0 ; 
}
