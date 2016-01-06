#ifdef COMPLEX
#include <boost/bind.hpp>
#endif



#ifdef COMPLEX
public:
    // general-ish ptree updating 
    // http://stackoverflow.com/questions/8154107/how-do-i-merge-update-a-boostproperty-treeptree 
    void complex_update(const boost::property_tree::ptree& pt);
protected:
    template<typename T>
    void  _traverse(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type& childPath, const boost::property_tree::ptree& child, T method);

    template<typename T>
    void traverse(const boost::property_tree::ptree &parent, T method);
    void merge(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type &childPath, const boost::property_tree::ptree &child);
#endif




#ifdef COMPLEX
// general soln not needed for simple ini format Bookmarks tree structure, 
// but for future more complex structures...
//
// http://stackoverflow.com/questions/8154107/how-do-i-merge-update-a-boostproperty-treeptree 
//
// The only limitation is that it is possible to have several nodes with the same
// path. Every one of them would be used, but only the last one will be merged.
//
// SO : restrict usage to unique key trees
//

void Bookmarks::complex_update(const boost::property_tree::ptree& pt) 
{
    traverse(pt, boost::bind(&Bookmarks::merge, this, _1, _2, _3));
}

template<typename T>
void Bookmarks::_traverse(
       const boost::property_tree::ptree &parent, 
       const boost::property_tree::ptree::path_type &childPath, 
       const boost::property_tree::ptree &child, 
       T method
       )
{
    method(parent, childPath, child);
    for(pt::ptree::const_iterator it=child.begin() ; it!=child.end() ;++it ) 
    {
        pt::ptree::path_type curPath = childPath / pt::ptree::path_type(it->first);
        _traverse(parent, curPath, it->second, method);
    }
}

template<typename T>
void Bookmarks::traverse(const boost::property_tree::ptree &parent, T method)
{
    _traverse(parent, "", parent, method);
}

void Bookmarks::merge(const boost::property_tree::ptree& parent, const boost::property_tree::ptree::path_type &childPath, const boost::property_tree::ptree &child) 
{
    LOG(info)<<"Bookmarks::merge " << childpath << " : " << child.data() ; 
    m_tree.put(childPath, child.data());
}    

#endif


