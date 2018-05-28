#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

#include "helpers.hpp"
#include "features_extractor.hpp"
#include "integral_images.hpp"
#include "training_adaboost.hpp"
#include "icf_naive_detector.hpp"
#include "icf_fpdw_detector.hpp"
#include "icf_multi_detector.hpp"
#include <boost/cstdint.hpp>
#include <boost/random.hpp>

using namespace std;
using namespace cv;










namespace{


void read_yuv_image(const char* inputname, vector<char>& outputdata)
{

    using namespace cv;
    ifstream fin;
    fin.open(inputname,ios_base::in|ios_base::binary);
    if(!fin.is_open())
    {
        throw runtime_error("Image Open Error");
    }


    outputdata.resize(640*480*2);
    fin.read(&outputdata[0],640*480*2);
    fin.close();

    return;
}

void get_y_from_yuv(const vector<char>& inputdata, vector<unsigned char>& outputdata)
{

    outputdata.resize(inputdata.size()/2);
    for(int i=0;i<outputdata.size();i++)
    {
        outputdata[i]=static_cast<unsigned char>(inputdata[i*2]);
    }
    return;
}

cv::Mat yuvRead(const char* inputname)
{
    vector<char> yuv;
    vector<unsigned char> ydata;
    read_yuv_image(inputname,yuv);
    get_y_from_yuv(yuv,ydata);
    cv::Mat temp(480,640,CV_8UC1,&ydata[0],640);
    cv::Mat retv;
    temp.copyTo(retv);
    return retv;
}

cv::Mat fakeMask(const cv::Mat& img)
{
    int white_y = 160;

    cv::Mat retv(cv::Size(img.cols,img.rows),CV_8UC1);
    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            retv.at<unsigned char>(i,j) = img.at<unsigned char>(i,j)>white_y?255:0;
        }
    }


    return retv;
}

}




int main(int argc, char** argv)
{
    if(argc < 7)
        throw std::runtime_error("Example input: ./main_train pos48.txt neg.txt 48 72 30000 goal");
    ICFHelp::getTrainingFlag(1);


    string positives_path(argv[1]);
    string negatives_path(argv[2]);
    int model_wid = atoi(argv[3]);
    int model_hei = atoi(argv[4]);
    int featurenum = atoi(argv[5]);
    string save_hint(argv[6]);


    ICFHelp::getModelWidth(model_wid);
    ICFHelp::getModelHeight(model_hei);
    ICFHelp::getRandFeatureNum(featurenum);
    ICFHelp::getShrinkFactor(2);
    ICFHelp::getChannel(1);



    cout<<"training"<<endl;
    TrainingAdaboost *training = TrainingAdaboost::getInstance();
    training->readPositive(positives_path);
    training->readBigNegative(negatives_path);
    training->cutRandomNegative(1000);

    int max_bootstrap_rounds = 4;
    size_t max_negs_per_img = 4;
    int max_model_stages = 40;
    for(int k=0;k<max_bootstrap_rounds;k++)
    {
        cout<<"\n\n---------------------------- training bootstrap round "<<k<<" ----------------------"<<endl;

        /// bootstrap false positive
        if(k!=0)
        {
            int num_add=0;
            int imgid=0;
            stringstream modelss;
            modelss<<"model_"<<save_hint<<"_"<<model_wid<<"_"<<k-1<<".proto.bin";
            string modelname = modelss.str();
            ICFNaiveDetector detector(modelname,0.7,8.0,16,1.0,1.0,0.0);
            ifstream testn(negatives_path.c_str());
            string name;
            std::getline(testn,name);
            while(!testn.eof())
            {
                cv::Mat img = cv::imread(name);
                detector.setImage(img);
                vector<ICFTrainData> toadd_negs = detector.compute();
                cout<<"bootstrap image after compute -----   ";
                cout<<"find fp: "<<toadd_negs.size()<<" -- ";
                toadd_negs.resize(min(max_negs_per_img,toadd_negs.size()),ICFTrainData(6000));
                num_add += training->addNegatives(toadd_negs);
                cout<<"bootstrap image "<<imgid++<<"  total num added: "<<num_add<<endl;
                std::getline(testn,name);
            }
        }

        if(true)//debug
        {
            cout<<"train with pos: "<<training->getPosNum()<<"----- neg: "<<training->getNegNum()<<endl;
        }

        training->initNewTrain();
        ICFDetector::StrongModel strongc;
        strongc.set_shrink_factor(ICFHelp::getShrinkFactor());
        strongc.set_model_width(ICFHelp::getModelWidth());
        strongc.set_model_height(ICFHelp::getModelHeight());
        int stage=0;
        while(stage<max_model_stages)
        {
            ICFDetector::WeakModel* weakc = strongc.add_weak_models();
            double errrate;
            *weakc = training->trainWeakModel(errrate);
            cout<<"cascade stagee: "<<stage<<"   errorrate: "<<errrate<<endl;
            stage++;
        }


        stringstream outss;
        outss<<"model_"<<save_hint<<"_"<<model_wid<<"_"<<k<<".proto.bin";
        fstream out(outss.str().c_str(), ios::out | ios::binary | ios::trunc);
        strongc.SerializeToOstream(&out);
        out.close();
    }

    cout<<"training end"<<endl;
    return 0;
}


