#include "knn_test.h"

using namespace std;
using namespace cv;
using namespace cv::ml;


void TestDigit::load_model(string model_path)
{
  this->kclassifier = Algorithm::load<KNearest>(model_path);
  cout << "success load model" << endl;
}

float TestDigit::inference(Mat& img)
{
  resize(img, img, Size(this->image_size, this->image_size), INTER_LINEAR);
  Mat flatten_vec = img.clone().reshape(0, 1);
  flatten_vec.convertTo(flatten_vec, CV_32F);
  float f = this->kclassifier->predict(flatten_vec);
  return f;
}