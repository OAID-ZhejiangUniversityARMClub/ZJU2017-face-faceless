#include <map>
#include <string>

#include "face_detector.hpp"

typedef std::map<std::string, detector_factory::creator> creator_map;
int side_size = 384;

static creator_map & get_registry(void)
{
   static creator_map * instance_ptr=new creator_map;

   return (*instance_ptr);
}


void detector_factory::register_creator(const std::string&name, detector_factory::creator& create_func)
{
   creator_map& registry=get_registry();

   registry[name]=create_func;
}


face_detector * detector_factory::create_face_detector(const std::string& name)
{
    std::vector<std::string> ret;

    creator_map& registry=get_registry();

    if(registry.find(name)==registry.end())
        return nullptr;

     creator func=registry[name];

    return func(name);
}

std::vector<std::string> detector_factory::list_detector(void)
{
 
    std::vector<std::string> ret;

    creator_map& registry=get_registry();

    creator_map::iterator it=registry.begin();
 
    while(it!=registry.end())
    {
        ret.push_back(it->first);
        it++;
    }

    return ret;
}
