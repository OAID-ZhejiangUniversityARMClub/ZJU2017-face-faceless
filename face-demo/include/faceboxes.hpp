#ifndef __FACEBOXES_H__
#define __FACEBOXES_H__

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#include "face_detector.hpp"
#include <string>

/*
* from:
*   https://github.com/AlfredXiangWu/face_verification_experiment
*/

using namespace caffe;

class faceboxes : public face_detector
{

public:

     faceboxes(const std::string name):face_detector(name),net_(nullptr){};

     cv::Mat preprocess(const cv::Mat& frame) ;
	 cv::Mat bordImg(const cv::Mat& frame);
	 cv::Mat getMean(const size_t& imageHeight, const size_t& imageWidth);

     int load_model(const std::string& model_dir);

	 void detect(cv::Mat& img, std::vector<face_box>& face_list);

     ~faceboxes(void);


private:

    Net<float> * net_;     

};

#endif

