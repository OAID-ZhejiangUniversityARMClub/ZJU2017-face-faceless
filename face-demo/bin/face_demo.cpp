#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <unistd.h>
#include <signal.h>

#include "mtcnn.hpp"
#include "face_align.hpp"
#include "feature_extractor.hpp"
#include "face_verify.hpp"
#include "face_mem_store.hpp"
#include "network_shell.hpp"
#include "face_detector.hpp"
#include "feature_extractor.hpp"
#include <queue>
#include "utils.hpp"

#ifndef MODEL_DIR
#define MODEL_DIR "./models"
#endif

volatile int quit_flag=0;

#define CMD_STATUS_IDEL 0
#define CMD_STATUS_PENDING  1
#define CMD_STATUS_RUN      2
#define CMD_STATUS_DONE     3

#define UNKNOWN_FACE_ID_MAX 1000

#define arm64_sync  asm volatile("dsb sy" : : : "memory")
#define x86_sync    asm volatile("mfence":::"memory")

#ifdef __x86_64__
#define mem_sync x86_sync
#else
#define mem_sync arm64_sync
#endif

struct shell_cmd_para
{
	volatile int cmd_status; /* idle, pending,run,done */

	std::string op;

	unsigned int face_id;  
	std::string name;

	shell_cmd_para(void) { cmd_status=CMD_STATUS_IDEL;}


};

typedef void (*shell_exec_func_t)(shell_cmd_para *);

std::map<std::string,shell_exec_func_t> shell_exec_table;


face_detector * p_detector;
feature_extractor * p_extractor;
face_verifier   * p_verifier;
face_mem_store * p_mem_store;
cv::Mat * p_cur_frame;
int current_frame_count=0;

int win_keep_limit=10;
int trace_pixels=10;
float similar_thre = 0.35;
int mini_object_nums = 100;
bool is_print_similar = true;
bool is_use_skin = false;
bool is_show_windows = false;

struct face_window
{
	face_box box;
	unsigned int face_id;
	unsigned int frame_seq;
	float center_x;
	float center_y;
	std::string name;
	char title[128];
	float max_similar;
};

std::vector<face_window*> face_win_list;

int get_new_unknown_face_id(void)
{
	static unsigned int current_id=0;

	return  (current_id++%UNKNOWN_FACE_ID_MAX);
}


shell_cmd_para * get_shell_cmd_para(void)
{
	static shell_cmd_para * p_para=new shell_cmd_para();

	return p_para;
}


void register_shell_executor(const std::string &name, shell_exec_func_t func)
{
	shell_exec_table[name]=func;
}

void execute_shell_command(shell_cmd_para * p_para)
{
	shell_exec_func_t func;

	std::map<std::string,shell_exec_func_t>::iterator it;

	it=shell_exec_table.find(p_para->op);

	if(it ==shell_exec_table.end())
	{
		std::cerr<<"Not such command: "<<p_para->op<<std::endl;
		p_para->cmd_status=CMD_STATUS_DONE;
		return;
	}

	p_para->cmd_status=CMD_STATUS_RUN;

	func=it->second;

	func(p_para);

	p_para->cmd_status=CMD_STATUS_DONE;
}

unsigned int get_new_registry_id(void)
{
	static unsigned int register_id=10000;

	register_id++;

	if(register_id==20000)
		register_id=10000;

	return register_id;
}

static void  exec_register_face_feature(shell_cmd_para * p_para)
{
	unsigned int face_id=p_para->face_id;
	unsigned int i;
	face_window * p_win;

	for(i=0;i<face_win_list.size();i++)
	{
		if(face_win_list[i]->face_id == face_id &&
				face_win_list[i]->frame_seq == current_frame_count)
			break;

	}

	if(i==face_win_list.size())
	{
		std::cout<<"cannot find face with id: "<<face_id<<std::endl;
		return;
	}

	p_win=face_win_list[i];

	/* extract feature first */

	face_info info;

	info.p_feature=(float *)malloc(128*sizeof(float));    //
	int x = p_win->box.x0>=0?p_win->box.x0:0, y = p_win->box.y0>=0?p_win->box.y0:0;
	int w = p_win->box.x1<(int)(*p_cur_frame).cols?p_win->box.x1-p_win->box.x0:(int)(*p_cur_frame).cols-p_win->box.x0-1;
	int h = p_win->box.y1<(int)(*p_cur_frame).rows?p_win->box.y1-p_win->box.y0:(int)(*p_cur_frame).rows-p_win->box.y0-1;
	cv::Mat aligned(*p_cur_frame, cv::Rect(x, y, w, h));;
	cv::resize(aligned, aligned, cv::Size(96, 112));
	/* align face */
	//get_aligned_face(*p_cur_frame,(float *)&p_win->box.landmark,5,128,aligned);

	/* get feature */
	p_extractor->extract_feature(aligned,info.p_feature);

	if(face_id<UNKNOWN_FACE_ID_MAX)
		info.face_id=get_new_registry_id();
	else
		info.face_id=face_id;

	info.name=p_para->name;
	info.feature_len=128;     //

	/* insert feature into mem db */

	p_mem_store->insert_new_record(info);

	/* insert feature into verifier */

	p_verifier->insert_feature(info.p_feature,info.face_id);    

}

void register_face_feature(int argc, char * argv[])
{
	int ret;
	char * username=NULL;
	int face_id=-1;

	optind=1;

	while((ret=getopt(argc,argv,"i:u:"))!=-1)
	{
		switch(ret)
		{
			case 'i':
				face_id=strtoul(optarg,NULL,10);
				break;
			case 'u':
				username=optarg;
				break;
			default:
				break;
		}

	}

	if(face_id<0 || username==NULL)
	{
		fprintf(stdout,"bad arguments\n");
		return ;
	}

	/* check if face_id is a registered one */

	face_info * p_info=p_mem_store->find_record(face_id);

	if(p_info && p_info->name != username)
	{
		fprintf(stdout,"do not support change name from %s to %s\n",
				p_info->name.c_str(),username);
		return ;
	}


	/* setup command para */

	shell_cmd_para * p_para=get_shell_cmd_para();

	p_para->op="reg";
	p_para->face_id=face_id;
	p_para->name=username;

	mem_sync;

	p_para->cmd_status=CMD_STATUS_PENDING;

}
/* list registered faces */

void list_registered_face_info(int argc, char * argv[])
{
	std::vector<face_info *> list;
	int n=p_mem_store->get_all_records(list);

	for(int i=0;i<n;i++)
	{
		face_info * p_info=list[i];

		printf("%-2d\t%d\t%s\n",i,p_info->face_id,p_info->name.c_str());
	}

	std::cout<<"total "<<n<<" faces registered"<<std::endl;
}


/* remove face feature */

void delete_face_feature(int argc, char * argv[])
{
	int ret;
	int face_id=-1;
	char * username=NULL;
	optind=1;

	while((ret=getopt(argc,argv,"i:u:"))!=-1)
	{
		switch(ret)
		{
			case 'i':
				face_id=strtoul(optarg,NULL,10);
				break;
			case 'u':
				username=optarg;
				break;
			default:
				break;
		}
	}

	if(face_id>=0 && username!=NULL)
	{
		fprintf(stdout,"cannot set face_id and name both at one time\n");
		return ;    
	}

	if((face_id<0) && (username==NULL))
	{
		fprintf(stdout,"bad arguments\n");
		return ;
	}

	/* setup cmd para */

	/* setup command para */

	shell_cmd_para * p_para=get_shell_cmd_para();

	p_para->op="park";

	mem_sync;

	p_para->cmd_status=CMD_STATUS_PENDING;

	while(p_para->cmd_status!=CMD_STATUS_RUN);
	/* cv thread is parking now */

	std::vector<face_info *> list;

	if(username)
	{
		p_mem_store->find_record(username,list);
	}
	else
	{
		face_info  * p=p_mem_store->find_record(face_id);

		if(p!=nullptr)
			list.push_back(p);
	}

	if(list.size()==0)
	{
		std::cout<<"No target face found"<<std::endl;
	}

	for(int i=0;i<list.size();i++)
	{
		face_info * p=list[i];
		face_id=p->face_id;

		p_verifier->remove_feature(face_id);
		p_mem_store->remove_record(face_id);

		/* change the name in face_win_list to unknown */

		for(int l=0;l<face_win_list.size();l++)
		{
			face_window * p_win=face_win_list[l];

			if(p_win->face_id==face_id)
			{
				p_win->face_id=get_new_unknown_face_id();
				p_win->name="unknown";
				sprintf(p_win->title,"%d %s",p_win->face_id,p_win->name.c_str());
			}
		}

	}

	std::cout<<"total "<<list.size()<<" face/feature deleted"<<std::endl;


	p_para->cmd_status=CMD_STATUS_DONE;
}

void change_face_id_name(int argc, char * argv[])
{
	int ret;
	optind=1;
	int face_id=-1;
	char * username=NULL;

	while((ret=getopt(argc,argv,"i:u:"))!=-1)
	{
		switch(ret)
		{
			case 'i':
				face_id=strtoul(optarg,NULL,10);
				break;
			case 'u':
				username=optarg;
				break;
			default:
				break;
		}

	}

	if(face_id<0 || username==NULL)
	{
		fprintf(stdout,"bad arguments\n");
		return ;
	}

	/* check if face_id is a registered one */

	face_info * p_info=p_mem_store->find_record(face_id);

	if(p_info ==nullptr)
	{
		fprintf(stdout,"No such face id: %d\n",face_id);
		return ;
	}

	if(p_info->name == username)
	{
		fprintf(stdout,"Nothing needs to do\n");
		return ;
	}

	/* setup command para */

	shell_cmd_para * p_para=get_shell_cmd_para();
	p_para->op="park";
	mem_sync;
	p_para->cmd_status=CMD_STATUS_PENDING;

	while(p_para->cmd_status!=CMD_STATUS_RUN);

	p_info->name=username;

	/* update win */
	for(int i=0;i<face_win_list.size();i++)
	{
		face_window * p_win=face_win_list[i];

		if(p_win->face_id == face_id)
		{
			p_win->name=p_info->name;
		}
	}

	p_para->cmd_status=CMD_STATUS_DONE;
}


void park_cv_thread(shell_cmd_para * p_para)
{
	while(p_para->cmd_status!=CMD_STATUS_DONE)
	{
		asm volatile("":::"memory");
	}

}

void exit_face_demo(int argc, char * argv[])
{
	/* it is too rude ... */
	quit_flag=1;
}

void set_some_para(int argc, char * argv[]){
	int ret;
	optind=1;

	while((ret=getopt(argc,argv,"h:m:p:s:k:w:"))!=-1)
	{
		switch(ret)
		{
			case 'h':
				similar_thre=atof(optarg);
				break;
			case 'm':
				mini_object_nums=atoi(optarg);
				break;
			case 'p':
				is_print_similar=(atoi(optarg)==0?false:true);
				break;
			case 's':
				side_size=atoi(optarg);
				break;
			case 'k':
				is_use_skin=(atoi(optarg)==0?false:true);
				break;
			case 'w':
				is_show_windows=(atoi(optarg)==0?false:true);
				break;
			default:
				break;
		}
	}
}

void list_some_para(int argc, char * argv[]){
	std::cout<<"similar_thre: "<<similar_thre<<std::endl;
	std::cout<<"mini_object_nums: "<<mini_object_nums<<std::endl;
	std::cout<<"is_print_similar: "<<is_print_similar<<std::endl;
	std::cout<<"side_size: "<<side_size<<std::endl;
	std::cout<<"is_use_skin: "<<is_use_skin<<std::endl;
	std::cout<<"is_show_windows: "<<is_show_windows<<std::endl;
}

void init_shell_cmd(void)
{
	/* this is for command executed in cv thread */
	register_shell_executor("reg", exec_register_face_feature);
	register_shell_executor("park", park_cv_thread);

	/* this for command executed in net shell thread */
	register_network_shell_cmd("reg",register_face_feature,"reg -i face_id -u name","register/update a face feature into system");

	register_network_shell_cmd("list",list_registered_face_info,"list","display info of all registered faces");

	register_network_shell_cmd("del",delete_face_feature,"del {-i face_id|-u name}","delete face features by face id or by name");

	register_network_shell_cmd("rename",change_face_id_name,"rename -i face_id -u new_name","rename the name of face feature by id");

	register_network_shell_cmd("exit",exit_face_demo,"exit","exit the demo");

	register_network_shell_cmd("set",set_some_para,"set {-h similar_thre | -m mini_object_nums}","set some parameters");

	register_network_shell_cmd("print",list_some_para,"print","print some parameters");

}



/***********************************************************************************/

void get_face_name_by_id(unsigned int face_id, std::string& name)
{
	face_info * p_info;

	p_info=p_mem_store->find_record(face_id);

	if(p_info==nullptr)
	{
		name="nullname";
	}
	else
	{
		name=p_info->name;
	}
}


void sig_user_interrupt(int sig, siginfo_t * info, void * arg)
{
	std::cout<<"User interrupt the program ...\n"<<std::endl;
	quit_flag=1;
}


void drop_aged_win(unsigned int frame_count)
{
	std::vector<face_window *>::iterator it=face_win_list.begin();

	while(it!=face_win_list.end())
	{
		if((*it)->frame_seq+win_keep_limit<frame_count)
		{
			delete (*it);
			face_win_list.erase(it);
		}
		else
			it++;
	}
}

face_window * get_face_id_name_by_position(face_box& box,unsigned int frame_seq)
{
	int found=0;
	float center_x=(box.x0+box.x1)/2;
	float center_y=(box.y0+box.y1)/2;
	face_window * p_win;

	std::vector<face_window *>::iterator it=face_win_list.begin();

	while (it!=face_win_list.end())
	{
		p_win=(*it);
		float offset_x=p_win->center_x-center_x;
		float offset_y=p_win->center_y-center_y;

		if((offset_x<trace_pixels) &&
				(offset_x>-trace_pixels) &&
				(offset_y<trace_pixels) &&
				(offset_y>-trace_pixels) &&
				(p_win->frame_seq+win_keep_limit)>=frame_seq)
		{
			found=1;
			break;
		}
		it++;
	}


	if(!found)
	{
		p_win=new face_window();
		p_win->name="unknown";
		p_win->face_id=get_new_unknown_face_id();
	}

	p_win->box=box;
	p_win->center_x=(box.x0+box.x1)/2;
	p_win->center_y=(box.y0+box.y1)/2;
	p_win->frame_seq=frame_seq;

	if(!found)
		face_win_list.push_back(p_win);

	return  p_win;

}

//void get_face_title(cv::Mat& frame,face_box& box,unsigned int frame_seq)
std::vector<face_window *> get_face_title(cv::Mat& frame, std::vector<face_box>& face_infos, unsigned int frame_seq){
	int face_id;
	float score;
	std::vector<face_window *> p_wins;
	for(int i=0;i<(int)face_infos.size();i++){
		face_window * p_win;
		p_win=get_face_id_name_by_position(face_infos[i], frame_seq);
		p_wins.push_back(p_win);
	}
	std::vector<std::vector<std::pair<int, float>>> res;
	int db_size = p_verifier->get_db_size();
	if(db_size >= mini_object_nums){
		for(int i=0;i<(int)face_infos.size();i++){
			face_box box = face_infos[i]; float feature[128];    // 
			int x = box.x0>=0?box.x0:0, y = box.y0>=0?box.y0:0;
			int w = box.x1<(int)frame.cols?box.x1-box.x0:(int)frame.cols-box.x0-1;
			int h = box.y1<(int)frame.rows?box.y1-box.y0:(int)frame.rows-box.y0-1;
			cv::Mat aligned(frame, cv::Rect(x, y, w, h));
			cv::resize(aligned, aligned, cv::Size(96, 112)); //128
			/* get feature */
			p_extractor->extract_feature(aligned, feature);
			/* search feature in db */
			std::vector<std::pair<int, float>> ret;
			p_verifier->search(feature, ret);
			res.push_back(ret);	
		}
	}
	std::vector<face_window *> p_wins_ret;
	std::vector<bool> rows((int)face_infos.size(), true), cols(db_size, true);
	for(int i=0; i<std::min((int)face_infos.size(), db_size); i++){
		float max_similar = -2.0; int max_face_id = -1, row = -1, col = -1;
		for(int j=0; j<(int)res.size(); j++){
			if(!rows[j])	continue;
			for(int k=0; k < db_size; k++){
				if(!cols[k])	continue;
				if(res[j][k].second >= max_similar){
					max_similar = res[j][k].second;
					max_face_id = res[j][k].first; row = j; col = k;
				}
			}
		}
		if(max_similar < similar_thre)	break;
		if(max_face_id != -1){
			face_window *p_win = new face_window();
			p_win->box = p_wins[row]->box;
			p_win->center_x = p_wins[row]->center_x;
			p_win->center_y = p_wins[row]->center_y;
			p_win->frame_seq = p_wins[row]->frame_seq;
			rows[row] = false; cols[col] = false;
			p_win->face_id = max_face_id;
			p_win->max_similar = max_similar;
			get_face_name_by_id(max_face_id, p_win->name);
			std::vector<face_info *> name_list;
			p_mem_store->find_record(p_win->name, name_list);
			for(int p=0; p<name_list.size(); p++){
				for(int q=0; q<db_size; q++)
					if(res[0][q].first == name_list[p]->face_id)	cols[q] = false;
			}
			if(is_print_similar)	sprintf(p_win->title,"%d %s %f",p_win->face_id, p_win->name.c_str(), p_win->max_similar);
			else	sprintf(p_win->title,"%d %s",p_win->face_id, p_win->name.c_str());
			p_wins_ret.push_back(p_win);
		}
	}
	for(int i=0;i<(int)p_wins.size();i++){
		if(rows[i]){
			face_window *p_win = new face_window();
			p_win->box = p_wins[i]->box;
			p_win->center_x = p_wins[i]->center_x;
			p_win->center_y = p_wins[i]->center_y;
			p_win->frame_seq = p_wins[i]->frame_seq;
			p_win->name = p_wins[i]->name;
			p_win->face_id = p_wins[i]->face_id;
			sprintf(p_win->title,"%d %s",p_win->face_id,p_win->name.c_str());
			p_wins_ret.push_back(p_win);
		}
	}
	return p_wins_ret;
}

void draw_box_and_title(cv::Mat& frame, face_box& box, char * title)
{
	float left,top;

	left=box.x0;
	top=box.y0-10;

	if(top<0)
	{
		top=box.y1+20;
	}

	cv::putText(frame,title,cv::Point(left,top),CV_FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);

	cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 2);
}

void skin_detect(cv::Mat& src, std::vector<face_box>& face_list){
	int ori_width = src.cols, ori_depth = src.rows;
	cv::blur( src, src, cv::Size(3,3) );
	cv::Mat hsv;
	cv::cvtColor(src, hsv, CV_BGR2HSV);
	cv::Mat bw, out;
	cv::inRange(hsv, cv::Scalar(0, 10, 60), cv::Scalar(20, 150, 255), bw);
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	cv::morphologyEx(bw, out, cv::MORPH_DILATE, element);
	std::vector<std::vector<cv::Point> > contour;
	std::vector<cv::Vec4i> hie;
	cv::findContours(out, contour, hie, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	for(int i=0;i<contour.size();i++){
		cv::Rect box = cv::boundingRect(cv::Mat(contour[i]));
		if(box.area() > 10000 && box.height/float(box.width) > 0.8){
			face_box tmp; 
			float y0 = box.tl().y-(box.br().y-box.tl().y)*0.15;
			float y1 = box.br().y+(box.br().y-box.tl().y)*0.03;
			float x0 = box.tl().x-(box.br().x-box.tl().x)*0.07;
			float x1 = box.br().x+(box.br().x-box.tl().x)*0.07;
			tmp.x0 = (x0>0?x0:0);
			tmp.y0 = (y0>0?y0:0);
			tmp.x1 = (x1<ori_width?x1:ori_width-1);
			tmp.y1 = (y1<ori_depth?y1:ori_depth-1);
			face_list.push_back(tmp);
		}
	}
	if(is_show_windows){
		cv::imshow("dst", bw);
		cv::imshow("morph", out);
		cv::waitKey(1);
	}
}

int main(int argc, char * argv[])
{
	const char * type="caffe";
	struct  sigaction sa;

	int res;

	while((res=getopt(argc,argv,"f:t:s:h:m:p:s:k:w:"))!=-1)
	{
		switch(res)
		{
			case 't':
				type=optarg;
				break;
			case 'h':
				similar_thre=atof(optarg);
				break;
			case 'm':
				mini_object_nums=atoi(optarg);
				break;
			case 'p':
				is_print_similar=(atoi(optarg)==0?false:true);
				break;
			case 's':
				side_size=atoi(optarg);
				break;
			case 'k':
				is_use_skin=(atoi(optarg)==0?false:true);
				break;
			case 'w':
				is_show_windows=(atoi(optarg)==0?false:true);
				break;
			default:
				break;
		}
	}

	sa.sa_sigaction=sig_user_interrupt;
	sa.sa_flags=SA_SIGINFO;
	sigemptyset(&sa.sa_mask);

	sigaction(SIGTERM,&sa,NULL);
	sigaction(SIGINT,&sa,NULL);



	std::string model_dir=MODEL_DIR;
	const std::string detector_name("faceboxes");
	p_detector=detector_factory::create_face_detector(detector_name);
	if(p_detector==nullptr)
	{
		std::cerr<<"create feature extractor: "<<detector_name<<" failed."<<std::endl;
		return 2;
	}
	p_detector->load_model(model_dir);

	/* alignment */

	/* extractor */
	const std::string extractor_name("sphereface");

	p_extractor=extractor_factory::create_feature_extractor(extractor_name);

	if(p_extractor==nullptr)
	{
		std::cerr<<"create feature extractor: "<<extractor_name<<" failed."<<std::endl;

		return 2;
	}

	p_extractor->load_model(model_dir);

	/* verifier*/

	p_verifier=get_face_verifier("cosine_distance");
	p_verifier->set_feature_len(p_extractor->get_feature_length());

	/* store */

	p_mem_store=new face_mem_store(128,10);

	shell_cmd_para * p_para=get_shell_cmd_para();

	init_network_shell();
	init_shell_cmd();
	create_network_shell_thread("face>",8080);


	cv::VideoCapture camera;
	//cv::VideoWriter out("result.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25.0, cv::Size(640, 480));

	camera.open(0);

	if(!camera.isOpened())
	{
		std::cerr<<"failed to open camera"<<std::endl;
		return 1;
	}


	cv::Mat frame;
	float ave = 0; unsigned long sum = 0;
	std::queue<unsigned long> que;
	int total_count = 0; float total_ave = 0;
	while(!quit_flag)
	{
		std::vector<face_box> face_info;

		camera.read(frame);

		current_frame_count++;

		unsigned long start_time=get_cur_time();
		if(!is_use_skin)	p_detector->detect(frame,face_info);
		else	skin_detect(frame,face_info);
		unsigned long detect_time=get_cur_time();

		unsigned long extract_start = get_cur_time();
		std::vector<face_window *> p_wins_ret = get_face_title(frame, face_info, current_frame_count);
		unsigned long extract_end=get_cur_time();

		if(p_para->cmd_status==CMD_STATUS_PENDING)
		{
			mem_sync;
			p_cur_frame=&frame;
			execute_shell_command(p_para);
		}

		for(unsigned int i=0;(int)i<p_wins_ret.size();i++)
		{
			draw_box_and_title(frame,p_wins_ret[i]->box,p_wins_ret[i]->title);
			delete p_wins_ret[i];
		}

		drop_aged_win(current_frame_count);

		unsigned long end_time=get_cur_time();
		
		/*---------calculate inference time------------*/
		unsigned long total_cost = end_time-start_time;
		que.push(total_cost);
		if (que.size() < 11){
			sum += que.back();
			ave = sum / float(que.size());
		}
		else{
			sum = sum + que.back() - que.front();
			ave = sum / float(10); que.pop();
		}
		int fps = ceil(1000000/ave);
		std::string fps_display = "FPS: "+std::to_string(fps);
		cv::putText(frame,fps_display,cv::Point(5,25),CV_FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 255, 0), 2);
		std::cerr<<"total detected: "<<face_info.size()<<" faces. used "<<total_cost/1000.<<" ms"<<std::endl;
		std::cerr<<"detect cost: " << (detect_time-start_time)/1000.<<" ms; extract cost: " <<(extract_end-extract_start)/1000.<<" ms"<<std::endl;
		std::cerr<<"average inference time: " << ave/1000.<<" ms"<<std::endl;
		cv::imshow("camera",frame);
		//out << frame;
		cv::waitKey(1);
		total_ave += ave/1000.;		total_count++;
	}
	camera.release();
	//out.release();
	cv::destroyAllWindows();
	std::cerr<<"program average inference time: " << total_ave/total_count<<" ms"<<std::endl;
	return 0;
}


