#include "knn_train.h"


using namespace std;
using namespace cv;
using namespace cv::ml;


void TrainDigit::train_val_split(Mat& img)
{
  vector<Mat> TrainData;
  vector<Mat> ValData;

  int cnt = 0;
  for (int r = 0; r < img.rows; r += this->image_size)
  {
    int gt = r / 100;
    for (int c = 0; c < img.cols; c += this->image_size)
    {
      Mat block_img = img(Range(r, min(r + this->image_size, img.rows)), Range(c, min(c + this->image_size, img.cols)));
      Mat flatten_vec = block_img.clone().reshape(0, 1);
      flatten_vec.convertTo(flatten_vec, CV_32F);
      if (c % (2*this->image_size) == 0) {
        TrainData.push_back(flatten_vec);
        this->TrainLabel.push_back((float) gt);
      } else {
        ValData.push_back(flatten_vec);
        this->ValLabel.push_back((float) gt);
        cnt += 1;
      }
    }
  }

  vconcat(TrainData, this->TrainData);
  vconcat(ValData, this->ValData);
}

void TrainDigit::train()
{
  this->kclassifier = KNearest::create();
  this->kclassifier->setIsClassifier(true);
  this->kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
  this->kclassifier->setDefaultK(5);
  Ptr<cv::ml::TrainData> KnnTrainData = TrainData::create(this->TrainData, ROW_SAMPLE, this->TrainLabel);
  this->kclassifier->train(KnnTrainData);
}

void TrainDigit::validation()
{
  Mat result;
  int correct_num = 0;
  for (int r = 0; r < ValLabel.rows; r += 1){
    Mat sample = this->ValData(Range(r, r+1), Range::all());
    float f = this->kclassifier->predict(sample);
    if(this->ValLabel.at<float>(r,0) - f == 0){
      correct_num += 1;
    } 
  }
  cout << correct_num << endl;
}

void TrainDigit::save_model(string out_model_path)
{
  this->kclassifier->save(out_model_path);
  cout << "success save model" << endl;
}
