#include <iostream>
#include <vector>
#include <string>

#include "fhog.h"
#include "fhog1.hpp"
#include <opencv2/opencv.hpp>


int main(int argc, char** argv)
{
	std::string img_path = "./../../test_sign.jpg";

    cv::Mat img = cv::imread(img_path);
	if (img.empty()) {
		std::cout << "Fail read " << img_path << std::endl;
	}

    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::Mat resize_img;
	cv::resize(img, resize_img, cv::Size(100,46));
	auto st_new_fhog = cv::getTickCount();

	FHOG fhogDescripter;
	for (int i = 0; i < 1; ++i)
	{
		fhogDescripter.static_Init(resize_img.size(), 2);
		cv::Mat feat;
		fhogDescripter.compute(resize_img, feat, 2);
	}
    
	auto ed_new_fhog = cv::getTickCount();
	double time_new_fhog = (ed_new_fhog - st_new_fhog) / cv::getTickFrequency();


	// IplImage img_0 = img;
	// auto st_old_fhog = cv::getTickCount();
	// CvLSVMFeatureMapCaskade** map = new CvLSVMFeatureMapCaskade*;
	// for (int i = 0; i < 200; ++i)
	// {
	// 	getFeatureMaps(&img_0, 4, map);
	// 	normalizeAndTruncate(*map, 0.2);
	// 	PCAFeatureMaps(*map);

	// 	freeFeatureMapObject(map);
	// }

	// auto ed_old_fhog = cv::getTickCount();
	// double time_old_fhog = (ed_old_fhog - st_old_fhog) / cv::getTickFrequency();
	// delete map;


	// cout << "Original FHOG:" << time_old_fhog << endl;
	std::cout << "New FHOG: " << time_new_fhog*1000 <<"ms"<< std::endl;
	std::cout << "fhogDescripter.imageSize :" << fhogDescripter.imageSize << std::endl;
	std::cout << "fhogDescripter.sz :" << fhogDescripter.sz << std::endl;
	std::cout << "fhogDescripter.map.size() :" << fhogDescripter.map.size() << std::endl;
	// cv::imshow("image", img);
	// cv::waitKey(0);
    return 0;
}