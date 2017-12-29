#include "feature_extractor.hpp"
#include "sphereface.hpp"

#include <exception>


int sphereface::load_model(const std::string& proto_model_dir)
{
	Caffe::set_mode(Caffe::CPU);

	try{
		net_=new Net<float>((proto_model_dir + "/sphereface_128_nobn.prototxt"), caffe::TEST);
		net_->CopyTrainedLayersFrom(proto_model_dir + "/sphereface_128_nobn.caffemodel");
	}

	catch(std::exception&e)
	{
		if(net_)
			delete net_;

		return -1;
	}
	cv::FileStorage fs((proto_model_dir + "/meanAndthresh.xml"), cv::FileStorage::READ);
	if(!fs.isOpened()){  
        std::cout<<"failed to open file meanAndthresh.xml "<<std::endl;  
        return -1;  
    } 
	fs["mean"] >> mean_;
	fs["thresh"] >> thresh;

	return 0;

}

sphereface:: ~sphereface(void)
{
	if(net_)
		delete net_;
}

cv::Mat sphereface::getMean(const size_t& imageHeight, const size_t& imageWidth){
	cv::Mat mean;
	const double meanValues[3] = {127.5, 127.5, 127.5};
	std::vector<cv::Mat> meanChannels;
	for(int i = 0; i < 3; i++){
		cv::Mat channel((int)imageHeight, (int)imageWidth, CV_32F, cv::Scalar(meanValues[i]));
		meanChannels.push_back(channel);
	}
	cv::merge(meanChannels, mean);
	return mean;
}

cv::Mat sphereface::preprocess(const cv::Mat& frame)
{
	cv::Mat preprocessed;
	frame.convertTo(preprocessed, CV_32F);
	size_t width = frame.cols, height = frame.rows;
	cv::Mat mean = getMean(height, width);
	cv::subtract(preprocessed, mean, preprocessed);
	cv::Mat bord_img = preprocessed / 128;
	return bord_img;
}


int sphereface::extract_feature(cv::Mat & img, float * feature)
{
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

	const Blob<float> * feature_blob=net_->blob_by_name("fc5/sphere").get();

	if(feature_blob->shape(1)!= feature_len_)
	{
		return -1;
	}

	const float * output_data=feature_blob->cpu_data();

	for(int i=0;i<feature_len_;i++)
	{
		feature[i]= output_data[i]; //- mean_.at<double>(i);
	} 


	return 0;

}

/******************/



class only_for_auto_register
{
public:
   only_for_auto_register(const std::string& name, extractor_factory::creator creator)
   {
      extractor_factory::register_creator(name,creator);
   } 
      
};

static feature_extractor * sphereface_creator(const std::string& name)
{
      return new sphereface(name);
}

static only_for_auto_register dummy_instance("sphereface",sphereface_creator);

