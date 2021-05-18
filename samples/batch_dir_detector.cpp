#include "class_timer.hpp"
#include "class_detector.h"

#include <memory>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <iostream>
#include <experimental/filesystem>
#include <fstream>

namespace fs = std::experimental::filesystem;

int main([[maybe_unused]] int argc, char **argv) {
    Config config_v4;
    config_v4.net_type = YOLOV4;
    config_v4.file_model_cfg = "../configs/yolov4.cfg";
    config_v4.file_model_weights = "../configs/yolov4.weights";
    config_v4.calibration_image_list_file_txt = "../configs/calibration_images.txt";

    char *p;
    Precision pres;
    long conv = strtol(argv[3], &p, 10);
    if (errno != 0 || *p != '\0' || conv > 2 || conv < 0) {
        // Put here the handling of the error, like exiting the program with
        std::cout << "Precision is incorrectly provided" << std::endl;
        return 0;
        // an error message
    } else {
        // No error
        pres = static_cast<Precision>(conv);
    }

    config_v4.inference_precison = pres; //INT8 FP16 FP32
    config_v4.detect_thresh = 0.5;

    std::unique_ptr<Detector> detector(new Detector());
    detector->init(config_v4);

    std::vector<BatchResult> batch_res;
    Timer timer;
    std::ofstream outdata;
    std::ifstream infile("../configs/class_list.txt");
    const int SIZE = 90;
    std::string classes[SIZE];


    int count = 0;
    while (count < SIZE && infile >> classes[count]) {
        count++;
    }

    std::string path = argv[1];
    std::list<std::string> dataset;
    for (const auto &entry : fs::directory_iterator(path)) {
        //prepare batch data
        //std::cout << entry.path() << std::endl;
        dataset.push_back(entry.path());
    }


    int batch_size;
    errno = 0;
    conv = strtol(argv[2], &p, 10);

    // Check for errors: e.g., the string does not represent an integer
    // or the integer is larger than int
    if (errno != 0 || *p != '\0' || conv > 100 || conv < 0) {
        // Put here the handling of the error, like exiting the program with
        std::cout << "Batch size is incorrectly provided" << std::endl;
        return 0;
        // an error message
    } else {
        // No error
        batch_size = static_cast<int>(conv);
    }

    while (!dataset.empty()) {
        std::vector<cv::Mat> batch_img;
        for (int i = 0; i < batch_size; ++i) {

            cv::Mat image0 = cv::imread(dataset.front(), cv::IMREAD_UNCHANGED);
            dataset.pop_front();
            cv::Mat temp0 = image0.clone();

            batch_img.push_back(temp0);
        }
        // open output file
        /*
        string dirpath(entry.path().parent_path());
        string out_file(entry.path().filename());
        while (true) {
            size_t pos = out_file.find(".jpg");
            if (pos != string::npos)
                out_file.replace(pos, 4, ".txt");
            else
                break;
        }
        string out_path = dirpath + "_ret/" + out_file;
        outdata.open(out_path);
        */
        //detect
        timer.reset();
        detector->detect(batch_img, batch_res);
        timer.out("detect");

        //disp
        for (int i = 0; i < batch_img.size(); ++i) {
            for (const auto &r : batch_res[i]) {
                //std::cout << "batch " << i << " id:" << classes[r.id] << " prob:" << r.prob << " rect:" << r.rect
                //          << std::endl;
                cv::rectangle(batch_img[i], r.rect, cv::Scalar(255, 0, 0), 2);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << "id:" << classes[r.id] << "  score:" << r.prob;
                cv::putText(batch_img[i], stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5,
                            cv::Scalar(0, 0, 255), 2);
                //outdata << r.id << " " << r.prob << " " << r.rect.x -r.rect.width/2 << " "  << r.rect.y -r.rect.height/2 << " " << r.rect.x + r.rect.width/2 << " "<< r.rect.y + r.rect.height/2 << " " << std::endl;
                //outdata << classes[r.id] << " " << r.prob << " " << r.rect.x << " " << r.rect.y << " "
                //        << r.rect.x + r.rect.width << " " << r.rect.y + r.rect.height << " " << endl;
            }
            //outdata.close();
            cv::imshow("image" + std::to_string(i), batch_img[i]);
        }
        cv::waitKey(0);
    }
}


