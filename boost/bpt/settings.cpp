// http://www.boost.org/doc/libs/1_58_0/libs/property_tree/examples/debug_settings.cpp

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
#include <string>
#include <set>
#include <exception>
#include <iostream>
namespace pt = boost::property_tree;

struct debug_settings
{
    std::string m_file;               
    int m_level;                      
    std::set<std::string> m_modules; 
    void load(const std::string &filename);
    void save(const std::string &filename);
};

void debug_settings::load(const std::string &filename)
{
    pt::ptree tree;
    pt::read_xml(filename, tree);
    m_file = tree.get<std::string>("debug.filename");
    m_level = tree.get("debug.level", 0);
    BOOST_FOREACH(pt::ptree::value_type &v, tree.get_child("debug.modules")) 
        m_modules.insert(v.second.data());
}

void debug_settings::save(const std::string &filename)
{
    pt::ptree tree;
    tree.put("debug.filename", m_file);
    tree.put("debug.level", m_level);
    BOOST_FOREACH(const std::string &name, m_modules)
        tree.add("debug.modules.module", name);
    pt::write_xml(filename, tree);
}

int main()
{
    try
    {
        debug_settings ds;
        ds.load("/tmp/bpt-demo.xml");  // bpt-;bpt-demo to write the file
        ds.save("/tmp/bpt-demo-out.xml");
        std::cout << "Success\n";
    }
    catch (std::exception &e)
    {
        std::cout << "Error: " << e.what() << "\n";
    }
    return 0;
}
