// http://www.boost.org/doc/libs/1_58_0/libs/property_tree/examples/debug_settings.cpp

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>
namespace pt = boost::property_tree;

struct bookmarks 
{
    void load(const std::string &filename);
    void save(const std::string &filename);
};

void bookmarks::load(const std::string &filename)
{
    pt::ptree tree;
    pt::read_ini(filename, tree);


    BOOST_FOREACH( boost::property_tree::ptree::value_type const& mk, tree.get_child("") ) 
    {
        std::string mkk = mk.first;
        std::cout << mkk << std::endl ;

        BOOST_FOREACH( boost::property_tree::ptree::value_type const& it, mk.second.get_child("") ) 
        {
            std::string itk = it.first;
            std::string itv = it.second.data();

            std::cout << "   " << itk << " : " << itv << std::endl ;
        }
    }

}

void bookmarks::save(const std::string &filename)
{
    pt::ptree tree;
    pt::write_ini(filename, tree);
}

int main()
{
    try
    {
        bookmarks mk;
        mk.load("bookmarks.ini");
        mk.save("bookmarks_out.ini");
        std::cout << "Success\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Error: " << e.what() << "\n";
    }
    return 0;
}
