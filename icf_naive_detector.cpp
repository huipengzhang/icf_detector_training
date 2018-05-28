#include "icf_naive_detector.hpp"
#include "helpers.hpp"
#include "features_extractor.hpp"
#include <fstream>
#include <stdexcept>
using namespace std;

ICFNaiveDetector::ICFNaiveDetector(string modelname, float minscale,float maxscale,int numscale,
                                   float scale1stridex, float scale1stridey, float threshold)
    :m_min_scale(minscale),
      m_max_scale(maxscale),
      m_scale_num(numscale),
      m_scale1_stride_x(scale1stridex),
      m_scale1_stride_y(scale1stridey),
      m_threshold(threshold)

{

    if(m_threshold <= ICFHelp::getCascadeThreshold())
        throw std::runtime_error("Threshold Less Than Cascade Threshold");

    fstream in(modelname.c_str(), ios::in | ios::binary);
    if (!m_scale1_model.ParseFromIstream(&in)) {
        throw std::runtime_error("Fail to Parse Model ProtoBuf File");
    }
    in.close();

    ICFHelp::getModelWidth(m_scale1_model.model_width());
    ICFHelp::getShrinkFactor(m_scale1_model.shrink_factor());
    m_scale1_model = ICFHelp::getScoreNormalize(m_scale1_model);
    m_detections.reserve(1000);

}

ICFNaiveDetector::~ICFNaiveDetector()
{

}


void ICFNaiveDetector::setImage(const cv::Mat &img)
{
    // FIXME waiting to add borders
    if(m_original_img.cols!=img.cols || m_original_img.rows!=img.rows)
        updateSearchParam(img.cols/ICFHelp::getShrinkFactor(),img.rows/ICFHelp::getShrinkFactor());

    if(img.channels()>1)
        cv::cvtColor(img,m_original_img,CV_RGB2GRAY);
    else
        m_original_img = img;
}

void ICFNaiveDetector::updateSearchParam(int w,int h)
{
    m_search_params.clear();


    float log_scale_min = log(m_min_scale);
    float log_scale_max = log(m_max_scale);
    float scale_log_step = (log_scale_max-log_scale_min)/max(1,m_scale_num-1);
    for(int i=0;i<m_scale_num;i++)
    {
        float detect_scale = std::exp(log_scale_min+scale_log_step*i);

        SearchParam temp;
        temp.mmodel = m_scale1_model;
        temp.mmodel_scale = 1.0;
        temp.minput_image_scale = 1.0/detect_scale;// shrink image to detect larger rects
        temp.mstride_x = max(1,static_cast<int>(round(m_scale1_stride_x)));
        temp.mstride_y = max(1,static_cast<int>(round(m_scale1_stride_y)));
        temp.mstart_x = 0;
        temp.mstart_y = 0;
        temp.mend_x = max(0,
                          static_cast<int>(floor(w*temp.minput_image_scale-ICFHelp::getModelWidth()/ICFHelp::getShrinkFactor()*temp.mmodel_scale)));//FIXME endx and endy should -1 or not?
        temp.mend_y = max(0,
                          static_cast<int>(floor(h*temp.minput_image_scale-ICFHelp::getModelHeight()/ICFHelp::getShrinkFactor()*temp.mmodel_scale)));

        m_search_params.push_back(temp);
    }
}



vector<ICFTrainData> ICFNaiveDetector::compute()
{
    m_detections.clear();


    float last_input_img_scale=-1;
    cv::Mat temp;
    vector<cv::Mat> temp_integral_vec;
    DetectResult temp_detection;


    vector<DetectResult> unrescaled_detects;
    vector<ICFTrainData>  retv;
    for(size_t i=0;i<m_search_params.size();i++)// different scale
    {
        const SearchParam& param = m_search_params.at(i);
        if(last_input_img_scale!=param.minput_image_scale)
        {
            cv::resize(m_original_img,temp,cv::Size(0,0),param.minput_image_scale,param.minput_image_scale,CV_INTER_AREA);
            m_integral_impl.setImage(temp.data,temp.cols,temp.rows,1);
            m_integral_impl.compute();
            temp_integral_vec = m_integral_impl.getIntegralImgs();
            last_input_img_scale = param.minput_image_scale;
        }

        double cascade_score = ICFHelp::getCascadeThreshold();
        vector< vector<double> > score_sum(param.mend_y-param.mstart_y+1, vector<double>(param.mend_x-param.mstart_x+1,0));


        for(int xx=param.mstart_x;xx<=param.mend_x;xx++) // for each pixel
        {
            for(int yy=param.mstart_y;yy<=param.mend_y;yy++)
            {
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

        // colloect detecition score > threshold
        for(int k=0;k<score_sum.size();k++)// for each row
        {
            const vector<double>& sv = score_sum.at(k);// one row, for each col
            for(int kk=0;kk<sv.size();kk++)
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

                    if(ICFHelp::getTrainingFlag())
                    {
                        DetectResult qq;
                        qq = temp_detection;
                        qq.mx = kk;
                        qq.my = k;
                        qq.mw = param.mmodel_scale*ICFHelp::getModelWidth();
                        qq.mh = param.mmodel_scale*ICFHelp::getModelHeight();
                        unrescaled_detects.push_back(qq);
                    }
                }
            }
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
    if(ICFHelp::getTrainingFlag())
    {
        FeaturesExtractor feaext(TrainingAdaboost::getInstance()->m_random_feas,0,0);
        ICFTrainData temptd(TrainingAdaboost::getInstance()->m_random_feas.size());
        cv::Mat tempimg;
        unrescaled_detects = nonMaximumSuppressionUnrescaled(unrescaled_detects);

        for(size_t i=0;i<unrescaled_detects.size();i++)
        {
            const DetectResult& dets = unrescaled_detects.at(i);
            cv::resize(m_original_img,tempimg,cv::Size(0,0),dets.mimgscale,dets.mimgscale,CV_INTER_AREA);
            feaext.setImage(tempimg.data,tempimg.cols,tempimg.rows,1);
            feaext.compute(dets.mx,dets.my);
            temptd.data_features = feaext.getFeatures();
            temptd.data_label = -1;//negs
            temptd.sum_score = 0;
            temptd.weight = 0;
            retv.push_back(temptd);
        }
    }

    return retv;
}

void ICFNaiveDetector::nonMaximumSuppression()
{
    std::vector<DetectResult> potentials = m_detections;
    sort(potentials.begin(),potentials.end(),DetectResultLarger);

    m_detections.clear();
    for(size_t i=0;i<potentials.size();i++)
    {
        const DetectResult& pdr = potentials.at(i);
        bool deleteit=false;
        for(size_t j=0;j<m_detections.size();j++)
        {
            const DetectResult& rdr = m_detections.at(j);
            if(isOverlapMin(pdr.mx,pdr.my,pdr.mx+pdr.mw,pdr.my+pdr.mh
                         ,rdr.mx,rdr.my,rdr.mw+rdr.mx,rdr.my+rdr.mh,
                         0.4))
            {
                deleteit=true;
                break;
            }
        }

        if(!deleteit)
            m_detections.push_back(pdr);
    }
}

std::vector<DetectResult> ICFNaiveDetector::nonMaximumSuppressionUnrescaled(const std::vector<DetectResult>& dets)
{

    std::vector<DetectResult> retv,temp = dets;
    random_shuffle(temp.begin(),temp.end());

    for(size_t i=0;i<temp.size();i++)
    {
        const DetectResult& pdr = temp.at(i);
        bool deleteit=false;
        for(size_t j=0;j<retv.size();j++)
        {
            const DetectResult& rdr = retv.at(j);
            float x00 = pdr.mx/pdr.mimgscale;
            float y00 = pdr.my/pdr.mimgscale;
            float x01 = (pdr.mx+pdr.mw)/pdr.mimgscale;
            float y01 = (pdr.my+pdr.mh)/pdr.mimgscale;

            float x10 = rdr.mx/rdr.mimgscale;
            float y10 = rdr.my/rdr.mimgscale;
            float x11 = (rdr.mx+rdr.mw)/rdr.mimgscale;
            float y11 = (rdr.my+rdr.mh)/rdr.mimgscale;


            if(isOverlapMin(x00,y00,x01,y01
                         ,x10,y10,x11,y11,
                         0.4))
            {
                deleteit=true;
                break;
            }
        }

        if(!deleteit)
            retv.push_back(pdr);
    }

    return retv;
}



const std::vector<DetectResult>& ICFNaiveDetector::getResults() const
{
    return m_detections;
}
