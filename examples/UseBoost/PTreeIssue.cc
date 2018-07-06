#include <boost/property_tree/ptree.hpp>

int main() {
    boost::property_tree::ptree b;
    b.push_back(std::make_pair("a", "b"));

    return 9;
}
