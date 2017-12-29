#ifndef __SPHEREFACE_H__
#define __SPHEREFACE_H__

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include "feature_extractor.hpp"
#include <string>

using namespace caffe;

class sphereface : public feature_extractor
{

public:

     sphereface(const std::string name):feature_extractor(name),net_(nullptr),feature_len_(128){};
	 cv::Mat getMean(const size_t& imageHeight, const size_t& imageWidth);
	 cv::Mat preprocess(const cv::Mat& frame);
     int get_feature_length(void) { return feature_len_;};

     int load_model(const std::string& model_dir);
	 void get_input_image_size(int& height, int& width) { height=112;width=96;}
     int extract_feature(cv::Mat & img, float * feature);

     ~sphereface(void);
	 double thresh;

private:

    Net<float> * net_;     
    int feature_len_;
	cv::Mat mean_;

};

#endif

