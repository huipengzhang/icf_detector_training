#include "icf_multi_detector.hpp"
#include "helpers.hpp"
#include <fstream>
#include <stdexcept>
using namespace std;

namespace
{
cv::Mat resizeImgMax(const cv::Mat& img, float rescale)
{
    int res = round(1.0/rescale);
    cv::Mat retv(img.rows/res,img.cols/res,CV_8UC1);
    for(int y=0;y<retv.rows;y++)
    {
        for(int x=0;x<retv.cols;x++)
        {
            retv.at<unsigned char>(y,x) =img.at<unsigned char>(y*res,x*res);
        }
    }

    return retv;
}
}



ICFMultiDetector::ICFMultiDetector(string multimodelname, float minscale,float maxscale,int numscale,
                                   float scale1stridex, float scale1stridey, float threshold,float cascade_threshold)
    :m_min_scale(minscale),
      m_max_scale(maxscale),
      m_scale_num(numscale),
      m_scale1_stride_x(scale1stridex),
      m_scale1_stride_y(scale1stridey),
      m_threshold(threshold),
      m_cascade_threshold(cascade_threshold)
{

    if(m_threshold <= m_cascade_threshold)
        throw std::runtime_error("Threshold Less Than Cascade Threshold");

    fstream in(multimodelname.c_str(), ios::in | ios::binary);
    if (!m_multi_scale_model.ParseFromIstream(&in)) {
        throw std::runtime_error("Fail to Parse Model ProtoBuf File");
    }
    in.close();

    for(int i=0;i<m_multi_scale_model.strong_models_size();i++)
    {
        ICFDetector::StrongModel* temp = m_multi_scale_model.mutable_strong_models(i);
        *temp = ICFHelp::getScoreNormalize(*temp);
    }

    ICFHelp::getShrinkFactor(m_multi_scale_model.shrink_factor());
    ICFHelp::getModelWidth(m_multi_scale_model.scale0_model_width());
    if(m_multi_scale_model.scale0_model_height()<0)
        ICFHelp::getModelHeight(m_multi_scale_model.scale0_model_width());
    else
        ICFHelp::getModelHeight(m_multi_scale_model.scale0_model_height());


    cout<<"MULTI --- Having "<<m_multi_scale_model.strong_models_size()
       <<" Strong Models and Scale1 Wid = "<<ICFHelp::getModelWidth()<<endl;

    m_detections.reserve(1000);
}

ICFMultiDetector::~ICFMultiDetector()
{

}


void ICFMultiDetector::setImage(const cv::Mat &img)
{
    // FIXME waiting to add borders
    if(m_original_img.cols!=img.cols || m_original_img.rows!=img.rows)
        updateSearchParam(img.cols/ICFHelp::getShrinkFactor(),img.rows/ICFHelp::getShrinkFactor());

    if(img.channels()>1)
        cv::cvtColor(img,m_original_img,CV_RGB2GRAY);
    else
        m_original_img = img;

    m_use_mask = false;
}

void ICFMultiDetector::setImage(const cv::Mat &img, const cv::Mat& mask)
{
    this->setImage(img);
    cv::resize(mask,m_shrinked_mask,
               cv::Size(img.cols/ICFHelp::getShrinkFactor(),img.rows/ICFHelp::getShrinkFactor()),
               0,0,cv::INTER_NEAREST);
    m_use_mask = true;
}

ICFDetector::StrongModel ICFMultiDetector::getProperModel(float log_scale)
{
    int modelind = min(max(0,static_cast<int>(round(log_scale))),m_multi_scale_model.strong_models_size()-1);

    const ICFDetector::StrongModel& chosen_model = m_multi_scale_model.strong_models(modelind);
    float relative_scale = pow(2,log_scale-modelind);
    ICFDetector::StrongModel retv = ICFHelp::getRescaledStrongModel(chosen_model,relative_scale);

    if(true)//debug
    {
        cout<<"MULTI --- model_chosen_scale: "<<pow(2,modelind)<<" model_up_sample_scale: "<<relative_scale
           <<" total_detect_scale: "<<pow(2,log_scale)<<endl;
    }
    return retv;
}

/// w,h is shrinked scale1 image's size
void ICFMultiDetector::updateSearchParam(int w,int h)
{
    m_search_params.clear();

    float log_scale_min = log(m_min_scale)/log(2.0);
    float log_scale_max = log(m_max_scale)/log(2.0);
    float scale_log_step = (log_scale_max-log_scale_min)/max(1,m_scale_num-1);
    for(int i=0;i<m_scale_num;i++)
    {
        float image_log_scale = log_scale_min+scale_log_step*i;
        SearchParam temp;
        temp.mmodel = getProperModel(image_log_scale);
        temp.mmodel_scale = pow(2,image_log_scale);
        temp.minput_image_scale = 1;
        temp.mstride_x = max(1,static_cast<int>(round(m_scale1_stride_x*temp.mmodel_scale)));
        temp.mstride_y = max(1,static_cast<int>(round(m_scale1_stride_y*temp.mmodel_scale)));
        temp.mstart_x = 0;
        temp.mstart_y = 0;
        temp.mend_x = max(0,static_cast<int>(floor(w*temp.minput_image_scale-ICFHelp::getModelWidth()/ICFHelp::getShrinkFactor()*temp.mmodel_scale)));//FIXME endx and endy should -1 or not?
        temp.mend_y = max(0,static_cast<int>(floor(h*temp.minput_image_scale-ICFHelp::getModelHeight()/ICFHelp::getShrinkFactor()*temp.mmodel_scale)));

        if(temp.mstart_x >= temp.mend_x && temp.mstart_y>=temp.mend_y)
        {
            break;
        }

        m_search_params.push_back(temp);
    }
}



void ICFMultiDetector::compute()
{
    m_detections.clear();


    float last_input_img_scale=-1;
    cv::Mat temp;
    cv::Mat tempmask;
    vector<cv::Mat> temp_integral_vec;
    DetectResult temp_detection;

    cv::TickMeter cvtimer;
    cvtimer.start();
    vector<float> timeee(10,0);
    for(size_t i=0;i<m_search_params.size();i++)// different scale
    {
        const SearchParam& param = m_search_params.at(i);
        //cout<<param.mstride_x<<" "<<param.mstride_y<<endl;

        if(last_input_img_scale!=param.minput_image_scale)
        {
            if(param.minput_image_scale!=1)
            {
                cv::resize(m_original_img,temp,cv::Size(0,0),param.minput_image_scale,param.minput_image_scale);                
                if(m_use_mask)
                {
//                    tempmask = resizeImgMax(m_shrinked_mask,param.minput_image_scale);
                    cv::resize(m_shrinked_mask,tempmask,cv::Size(0,0)
                           ,param.minput_image_scale
                           ,param.minput_image_scale
                           ,cv::INTER_NEAREST);
                }
            }
            else
            {
                m_original_img.copyTo(temp);
                if(m_use_mask)
                {
                    m_shrinked_mask.copyTo(tempmask);
                }
            }

//            stringstream a;
//            a<<"temp"<<i<<".png";
//            cv::imwrite(a.str().c_str(),tempmask);

            m_integral_impl.setImage(temp.data,temp.cols,temp.rows,1);
            m_integral_impl.compute();
            temp_integral_vec = m_integral_impl.getIntegralImgs();
            last_input_img_scale = param.minput_image_scale;

            if(true)//debug
            {
                //cout<<"MULTI --- doing resize once"<<endl;
                cvtimer.stop();
                //cout<<"MULTI --- resize using "<<cvtimer.getTimeMilli()<<" ms"<<endl;
                timeee[0]+=cvtimer.getTimeMilli();
                cvtimer.reset();
                cvtimer.start();
            }
        }
        else
        {
            if(false)//debug
            {
                cout<<"MULTI --- skip resize once"<<endl;
            }
        }

        double cascade_score = m_cascade_threshold;
        vector< vector<double> > score_sum(param.mend_y-param.mstart_y+1, vector<double>(param.mend_x-param.mstart_x+1,0));

//        cv::namedWindow("Mask");
//        cv::imshow("Mask",tempmask);
//        cv::waitKey(0);


        int maskadd_x = round(param.mmodel_scale*ICFHelp::getModelWidth()/ICFHelp::getShrinkFactor()/2);
        int maskadd_y = round(param.mmodel_scale*ICFHelp::getModelHeight()/ICFHelp::getShrinkFactor()/2);
        for(int xx=param.mstart_x;xx<=param.mend_x;xx+=param.mstride_x) // for each pixel
        {
            for(int yy=param.mstart_y;yy<=param.mend_y;yy+=param.mstride_y)
            {

                if(m_use_mask)
                {
                    if(tempmask.at<unsigned char>(yy+maskadd_y,xx+maskadd_x) < 128)
                    {
                        score_sum[yy-param.mstart_y][xx-param.mstart_x] = -1e10;
                        continue;
                    }
                }

                for(int j=0;j<param.mmodel.weak_models_size();j++) // different stages
                {

                    score_sum[yy-param.mstart_y][xx-param.mstart_x] += ICFHelp::weakModelJudge(param.mmodel.weak_models(j),
                                                                                               temp_integral_vec,
                                                                                               xx,yy);
                    if(score_sum[yy-param.mstart_y][xx-param.mstart_x] < cascade_score)
                        break;
                }
            }
        }



        if(true)
        {
            cvtimer.stop();
            //cout<<"MULTI --- model judge using "<<cvtimer.getTimeMilli()<<" ms"<<endl;
            timeee[1]+=cvtimer.getTimeMilli();
            cvtimer.reset();
            cvtimer.start();
        }

        // collect detecition score > threshold
        for(int k=0;k<score_sum.size();k+=param.mstride_y)// for each row
        {
            const vector<double>& sv = score_sum.at(k);// one row
            for(int kk=0;kk<sv.size();kk+=param.mstride_x)
            {
                if(sv[kk]>m_threshold)
                {
                    temp_detection.mx = static_cast<int>(round(kk/param.minput_image_scale*ICFHelp::getShrinkFactor()));
                    temp_detection.my = static_cast<int>(round(k/param.minput_image_scale*ICFHelp::getShrinkFactor()));
                    temp_detection.mw = static_cast<int>(round(param.mmodel_scale/param.minput_image_scale*ICFHelp::getModelWidth()));
                    temp_detection.mh = static_cast<int>(round(param.mmodel_scale/param.minput_image_scale*ICFHelp::getModelHeight()));

                    temp_detection.mscore = sv[kk];
                    temp_detection.mimgscale = param.minput_image_scale;
                    temp_detection.mmodelscale = param.mmodel_scale;

                    m_detections.push_back(temp_detection);
                }
            }
        }
        if(true)//debug
        {
            cvtimer.stop();
            //cout<<"MULTI --- collect detections using "<<cvtimer.getTimeMilli()<<" ms"<<endl;
            timeee[2]+=cvtimer.getTimeMilli();
            cvtimer.reset();
            cvtimer.start();
        }


        if(false)//debug
        {
            ofstream output("output.txt");
            for(int k=0;k<score_sum.size();k++)// for each row
            {
                const vector<double>& sv = score_sum.at(k);// one row
                for(int kk=0;kk<sv.size();kk++)
                {
                    output<<sv[kk]<<" ";
                }
                output<<endl;
            }
            output.close();
            throw std::runtime_error("Debug To Stop");
        }

    }

    nonMaximumSuppression();
    if(true)
    {
        cvtimer.stop();
        cout<<"MULTI --- NMS using "<<cvtimer.getTimeMilli()<<" ms"<<endl;
        timeee[3]+=cvtimer.getTimeMilli();
        cvtimer.reset();
        cvtimer.start();


        cout<<"MULTI --- ";
        for(int i=0;i<=3;i++)
        {
            cout<<timeee[i]<<"ms --- ";
        }
        cout<<endl;
    }



}

void ICFMultiDetector::nonMaximumSuppression()
{
    std::vector<DetectResult> potentials = m_detections;
    sort(potentials.begin(),potentials.end(),DetectResultLarger);

    m_detections.clear();
    for(size_t i=0;i<potentials.size();i++)
    {
        const DetectResult& pdr = potentials.at(i);
        int deleteit=0; // 0: not overlap, 1: overlap and delete potential, 2: overlap and replace detections
        int ti=-1;
        for(size_t j=0;j<m_detections.size();j++)
        {
            const DetectResult& rdr = m_detections.at(j);
            if(isOverlapMin(pdr.mx,pdr.my,pdr.mx+pdr.mw,pdr.my+pdr.mh
                         ,rdr.mx,rdr.my,rdr.mw+rdr.mx,rdr.my+rdr.mh,
                         0.3))
            {
                if(false && rdr.mw < pdr.mw)
                {
                    ti = j;
                    deleteit = 2;
                }
                else
                {
                    deleteit = 1;
                }

                break;
            }
        }

        switch(deleteit)
        {
        case 0:
            m_detections.push_back(pdr);
            break;
        case 1:
            break;
        case 2:
            m_detections[ti] = pdr;
            break;
        }
    }
}

const std::vector<DetectResult>& ICFMultiDetector::getResults() const
{
    return m_detections;
}
