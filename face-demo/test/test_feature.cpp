#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "feature_extractor.hpp"

#include "utils.hpp"

#ifndef MODEL_DIR
#define MODEL_DIR "./models"
#endif

int main(int argc, char * argv[])
{
	const char * type="mxnet";
	const char * fname="./test.jpg";
	int save_crop=0;

	int res;

	while((res=getopt(argc,argv,"f:t:s"))!=-1)
	{
		switch(res)
		{
			case 'f':
				fname=optarg;
				break;
			case 't':
				type=optarg;
				break;
			case 's':
				save_crop=1;
				break;
			default:
				break;
		}
	}



	cv::Mat frame = cv::imread(fname);

	if(!frame.data)
	{
		std::cerr<<"failed to read image file: "<<fname<<std::endl;
		return 1;
	}

	std::string model_dir=MODEL_DIR;

	feature_extractor * p_extractor;
	const std::string extractor_name("sphereface");

	p_extractor=extractor_factory::create_feature_extractor(extractor_name);

	if(p_extractor==nullptr)
	{
		std::cerr<<"create feature extractor: "<<extractor_name<<" failed."<<std::endl;

		return 2;
	}

	p_extractor->load_model(model_dir);
	float feature[256];

    unsigned long start_time=get_cur_time();

	p_extractor->extract_feature(frame,feature);

    unsigned long end_time=get_cur_time();


	for(unsigned int i=0;i<256;i++)
	{
		
		if(i != 255)	std::cout << feature[i] << " ";
		else	std::cout << feature[i] << std::endl;
	}

	std::cout<<"used "<<(end_time-start_time)<<" us"<<std::endl;

	return 0;
}

