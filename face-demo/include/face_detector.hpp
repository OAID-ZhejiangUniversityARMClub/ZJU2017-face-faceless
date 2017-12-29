#ifndef __FACE_DETECTOR_H__
#define __FACE_DETECTOR_H__

#include <string>
#include <opencv2/opencv.hpp>

#include "mtcnn.hpp"

extern int side_size;

class face_detector
{
public:
    virtual int load_model(const std::string& model_dir)=0;

    virtual void detect(cv::Mat& img, std::vector<face_box>& face_list)=0;

    virtual ~face_detector(void){};

    face_detector(const std::string&  name): name_(name){}

private:
   std::string name_;
   

};

class detector_factory
{
public:
    typedef face_detector * (*creator)(const std::string& name);

    static void register_creator(const std::string& name,creator& create_func);
    static face_detector * create_face_detector(const std::string& name);
    static std::vector<std::string> list_detector(void);
       
private:
    detector_factory(){};
};

#endif

