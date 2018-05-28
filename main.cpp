/**
  *
  *
  *  --------------------------   Currently not used now. ------------------------
  *
  *
  *
  *
  *
  *
  *
  *
  */






#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector>

#include "helpers.hpp"
#include "features_extractor.hpp"
#include "integral_images.hpp"
#include <boost/cstdint.hpp>
#include <boost/random.hpp>

using namespace std;
using namespace cv;



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

//bool leafmodel_less(const ICFDetector::LeafModel& a, const ICFDetector::LeafModel& b)
//{
//    if(a.channel_index()!=b.channel_index())
//    {
//        return a.channel_index()<b.channel_index();
//    }
//    else if(a.x0()!=b.x0())
//    {
//        return a.x0()<b.x0();
//    }
//    else if(a.y0()!=b.y0())
//    {
//        return a.y0()<b.y0();
//    }
//    else if(a.x1()!=b.x1())
//    {
//        return a.x1()<b.x1();
//    }
//    else if(a.y1()!=b.y1())
//    {
//        return a.y1()<b.y1();
//    }

//    return false;
//}






namespace{





}




int main(int argc, char** argv)
{
    int jumpid;
    if(argc<3)
        jumpid=0;
    else
        jumpid = atoi(argv[2]);

    ifstream f_img_list(argv[1]);
    string path;


    /// GUI init
    namedWindow( "Source", 1 );
    namedWindow( "Gradient", WINDOW_NORMAL );

    //
    vector<ICFDetector::LeafModel> random_leaf;
    ICFHelp::generate_random_leafs(random_leaf,1);
    FeaturesExtractor fea_extract(random_leaf,0,0);

    //
    IntegralImages big_img_processor;


    int count = 0;
    while(true)
    {

        std::getline(f_img_list,path);
        cout<<count++<<endl;
        if(jumpid>0)
        {
            jumpid--;
            cout<<"Jump frame"<<endl;
            continue;
        }

        Mat src, dst, color_dst;

        vector<unsigned char> src_y;
        vector<char> src_data;
        read_yuv_image(path.c_str(),src_data);
        get_y_from_yuv(src_data,src_y);

        src = Mat(480,640,CV_8UC1,&src_y[0]);
        dst = src;

        fea_extract.setImage(&src_y[0],640,480,1);
        fea_extract.compute();
        const vector<double>& tfea = fea_extract.getFeatures();

        imshow( "Source", src );
//        imshow( "Detected Lines", color_dst );

        char c = waitKey(0);
        if(c=='c')
        {
            break;
        }
    }


    return 0;
}

//int main(int argc, char** argv)
//{

//    int jumpid;
//    if(argc<3)
//        jumpid=0;
//    else
//        jumpid = atoi(argv[2]);

//    using namespace cv;
//    ifstream f_img_list(argv[1]);
//    string path;


//    /// GUI init
//    namedWindow( "Source", 1 );
//    namedWindow( "Detected Lines", WINDOW_NORMAL );


//    FeaIntegralGradient fea_extract;

//    int count = 0;
//    while(true)
//    {

//        std::getline(f_img_list,path);
//        cout<<count++<<endl;
//        if(jumpid>0)
//        {
//            jumpid--;
//            cout<<"Jump frame"<<endl;
//            continue;
//        }


//        ifstream fin;
//        fin.open(path.c_str(),ios_base::in|ios_base::binary);
//        cout<<"Open Succeed: "<<fin.is_open()<<endl;

//        Mat src, dst, color_dst;
//        vector<unsigned char> src_y;
//        vector<char> src_data;
//        src_y.resize(640*480);
//        src_data.resize(640*480*2);
//        fin.read(&src_data[0],640*480*2);
//        fin.close();

//        for(int i=0;i<src_y.size();i++)
//        {
//            src_y[i]=static_cast<unsigned char>(src_data[i*2]);
//        }
//        src = Mat(480,640,CV_8UC1,&src_y[0]);
//        dst = src;

//        fea_extract.setImage(&src_y[0],640,480,1);
//        fea_extract.compute();
//        dst = fea_extract.getGradientImgs();



//        cvtColor( dst, color_dst, CV_GRAY2BGR );


//        imshow( "Source", src );
//        imshow( "Detected Lines", color_dst );

//        char c = waitKey(0);
//        if(c=='c')
//        {
//            break;
//        }
//    }
//    return 0;
//}
