#include "face_detector.hpp"
#include "faceboxes.hpp"

#include <exception>


int faceboxes::load_model(const std::string& proto_model_dir)
{
	Caffe::set_mode(Caffe::CPU);

	try{
		net_=new Net<float>((proto_model_dir + "/m10-final-3_324x324_norm_32-40-48-96-144_nobn.prototxt"), caffe::TEST);
		net_->CopyTrainedLayersFrom(proto_model_dir + "/m10-final-3_324x324_norm_32-40-48-96-144_nobn.caffemodel");
	}

	catch(std::exception&e)
	{
		if(net_)
			delete net_;

		return -1;
	}

	return 0;

}

faceboxes:: ~faceboxes(void)
{
	if(net_)
		delete net_;
}

cv::Mat faceboxes::getMean(const size_t& imageHeight, const size_t& imageWidth){
	cv::Mat mean;
	const int meanValues[3] = {104, 117, 123};
	std::vector<cv::Mat> meanChannels;
	for(int i = 0; i < 3; i++){
		cv::Mat channel((int)imageHeight, (int)imageWidth, CV_32F, cv::Scalar(meanValues[i]));
		meanChannels.push_back(channel);
	}
	cv::merge(meanChannels, mean);
	return mean;
}

cv::Mat faceboxes::bordImg(const cv::Mat& frame){
	cv::Mat ret;
	int width = frame.cols, height = frame.rows;
	int bigger_side = width>height?width:height;
	//std::cerr << "side_size: " << float(side_size) << std::endl;
	float scale = float(side_size) / bigger_side;
	int new_width = int(width*scale), new_height = int(height*scale);
	cv::resize(frame, ret, cv::Size(new_width, new_height));
	return ret;
}

cv::Mat faceboxes::preprocess(const cv::Mat& frame)
{
	cv::Mat preprocessed;
	frame.convertTo(preprocessed, CV_32F);
	size_t width = frame.cols, height = frame.rows;
	cv::Mat mean = getMean(height, width);
	cv::subtract(preprocessed, mean, preprocessed);
	cv::Mat bord_img = bordImg(preprocessed);
	return bord_img;
}


void faceboxes::detect(cv::Mat& img, std::vector<face_box>& face_list){
	int ori_width = img.cols, ori_depth = img.rows;
	cv::Mat pre_image = preprocess(img);
	Blob<float>* input_blob = net_->input_blobs()[0];
	int pre_width = (int)pre_image.cols, pre_depth = (int)pre_image.rows, pre_chan = (int)pre_image.channels();
	input_blob->Reshape(1, pre_chan, pre_depth, pre_width);
	net_->Reshape();
	std::vector<cv::Mat> input_channels;
	int width = input_blob->width();  int height = input_blob->height();
	float * input_data=input_blob->mutable_cpu_data();
	for (int i = 0; i < input_blob->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += width * height;
	}
	cv::split(pre_image, input_channels);

	net_->Forward();

	/* get output*/

	Blob<float>* result_blob = net_->output_blobs()[0];
	const float* result = result_blob->cpu_data();
	const int num_det = result_blob->height();
	std::vector<std::vector<float> > detections;
	for (int k = 0; k < num_det; ++k) {
		if (result[0] == -1) {      // Skip invalid detection.
			result += 7;
			continue;
		}
		std::vector<float> detection(result, result + 7);
		face_box tmp; 
		float y0 = detection[4]-(detection[6]-detection[4])*0.5;
		float y1 = detection[6]+(detection[6]-detection[4])*0.1;
		float x0 = detection[3]-(detection[5]-detection[3])*0.07;
		float x1 = detection[5]+(detection[5]-detection[3])*0.07;
		tmp.x0 = (x0>0?x0*ori_width:0);
		tmp.y0 = (y0>0?y0*ori_depth:0);
		tmp.x1 = (x1<1?x1*ori_width:ori_width-1);
		tmp.y1 = (y1<1?y1*ori_depth:ori_depth-1);
		tmp.score = detection[2];
		if(tmp.score > 0)	face_list.push_back(tmp);
		result += 7;
	}
	return;
}


/******************/



class only_for_auto_register_fb
{
public:
   only_for_auto_register_fb(const std::string& name, detector_factory::creator creator)
   {
      detector_factory::register_creator(name,creator);
   } 
      
};

static face_detector * faceboxes_creator(const std::string& name)
{
      return new faceboxes(name);
}

static only_for_auto_register_fb dummy_instance("faceboxes",faceboxes_creator);

