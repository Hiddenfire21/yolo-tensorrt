#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <thread>

#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>
using namespace std;
namespace fs = std::experimental::filesystem;

int main(int argc, char** argv)
{
	Config config_v4;
	config_v4.net_type = YOLOV4;
	config_v4.file_model_cfg = "../configs/yolov4.cfg";
	config_v4.file_model_weights = "../configs/yolov4.weights";
	config_v4.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v4.inference_precison = FP32; //INT8 FP16 FP32
	config_v4.detect_thresh = 0.5;

	std::unique_ptr<Detector> detector(new Detector());
	detector->init(config_v4);

	std::vector<BatchResult> batch_res;
	Timer timer;
	ofstream outdata;
	ifstream infile("../configs/class_list.txt");
	const int SIZE = 90;
	string classes[SIZE];
	

	int count = 0;
	while ( count < SIZE && infile >> classes[count] )
	{
	    count++;
    }	
    
    std::string path = argv[1];
    for (const auto & entry : fs::directory_iterator(path))
	{
		//prepare batch data
		std::cout << entry.path() << std::endl;
		std::vector<cv::Mat> batch_img;
		cv::Mat image0 = cv::imread( entry.path(), cv::IMREAD_UNCHANGED);
		cv::Mat temp0 = image0.clone();

		batch_img.push_back(temp0);
		
		// open output file
		string dirpath(entry.path().parent_path());
		string out_file(entry.path().filename());
		while (true)
		{
		    size_t pos = out_file.find(".jpg");
		    if (pos != string::npos)
		        out_file.replace(pos, 4, ".txt");
	        else
	            break;
        }
        string out_path = dirpath + "_ret/" + out_file;
	    outdata.open(out_path);
		//detect
		timer.reset();
		detector->detect(batch_img, batch_res);
		timer.out("detect");

		//disp
		for (int i=0;i<batch_img.size();++i)
		{
			for (const auto &r : batch_res[i])
			{
				std::cout <<"batch "<<i<< " id:" << classes[r.id] << " prob:" << r.prob << " rect:" << r.rect << std::endl;
				cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
				std::stringstream stream;
				stream << std::fixed << std::setprecision(2) << "id:" << classes[r.id] << "  score:" << r.prob;
				cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
				//outdata << r.id << " " << r.prob << " " << r.rect.x -r.rect.width/2 << " "  << r.rect.y -r.rect.height/2 << " " << r.rect.x + r.rect.width/2 << " "<< r.rect.y + r.rect.height/2 << " " << std::endl;
				outdata << classes[r.id] << " " << r.prob << " " << r.rect.x << " "  << r.rect.y << " " << r.rect.x + r.rect.width << " "<< r.rect.y + r.rect.height << " " << endl;
			}
			outdata.close();
			cv::imshow("image"+std::to_string(i), batch_img[i]);
		}
		cv::waitKey(10);
	}
}
