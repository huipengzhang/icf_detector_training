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


    bool test = (argc>=3);
    if(!test)
    {
        cout<<"Example Input: ./main_test model_multiscale.proto.bin test_images.txt false";
        throw runtime_error("Input Arg Num Not Compatible");
    }

    bool need_white_mask = true;
    if(argc==4 && argv[3]=="false")
    {
        need_white_mask = false;
    }

    if(test)
    {
        ICFHelp::getChannel(1);// set to integral y channel

        cv::namedWindow("Debug",WINDOW_AUTOSIZE);
        string model_name = string(argv[1]);
        string test_images_list_name = string(argv[2]);

        cout<<"Input Model Name: "<<model_name<<endl;
        ICFMultiDetector me(model_name,1,4,40,1,1,0.1,-0.1);

        ifstream testn(test_images_list_name.c_str());
        string name;
        getline(testn,name);
        while(!testn.eof())
        {

            cv::Mat img;
            if(name.find(".yuv")!=std::string::npos)
            {
                img = yuvRead(name.c_str());
            }
            else
            {
                img = cv::imread(name);
            }


            if(need_white_mask)
            {
                cv::Mat mask=fakeMask(img);
                me.setImage(img,mask);
            }
            else
            {
                me.setImage(img);
            }
            me.compute();

            if(img.channels()==1)
                cv::cvtColor(img,img,CV_GRAY2BGR);

            vector<DetectResult> des = me.getResults();
            if(des.empty())
                cout<<"0 "<<name<<endl;
            else
            {
                float max_score = -1e5;
                float min_score = 1e5;
                for(int k=0;k<des.size();k++)
                {
                    const DetectResult& p = des[k];
                    if(p.mscore<min_score)
                        min_score = p.mscore;
                    if(p.mscore>max_score)
                        max_score = p.mscore;

                }

                for(int k=0;k<des.size();k++)
                {
                    cout<<name<<" "<<des.size()<<" (";
                    cout<<des[k].mx<<" "<<des[k].my<<" "<<des[k].mw<<" "<<des[k].mh<<") "
                       <<des[k].mmodelscale<<" "<<des[k].mscore<<endl;
                    cv::rectangle(img,Rect(des[k].mx,des[k].my,des[k].mw,des[k].mh),
                                  Scalar(0,0,(des[k].mscore-min_score)/(max_score-min_score)*255.0),2);
                }
            }
            cv::imshow("Debug",img);
            cv::waitKey(0);
            getline(testn,name);
        }
    }
    cout<<"test end"<<endl;

    return 0;
}


