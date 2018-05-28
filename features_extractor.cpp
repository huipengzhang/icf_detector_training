#include "features_extractor.hpp"
#include <stdexcept>
#include "integral_images.hpp"

using namespace std;

namespace{

}

FeaturesExtractor::FeaturesExtractor(const std::vector<ICFDetector::LeafModel>& leafs,
                                                   int shiftx,int shifty)
    :m_shiftx(shiftx),
      m_shifty(shifty),
      m_integral_img_processor(new IntegralImages)
{
    initFeatureInfo(leafs);
}

FeaturesExtractor::~FeaturesExtractor()
{

}

void FeaturesExtractor::initFeatureInfo(const std::vector<ICFDetector::LeafModel>& leafs)
{
    if(!leafs.empty())
    {
        m_fea_info = leafs;
        for(size_t i=0;i<m_fea_info.size();i++)
        {
            m_fea_info[i].set_x0(m_fea_info[i].x0()+m_shiftx);
            m_fea_info[i].set_x1(m_fea_info[i].x1()+m_shiftx);
            m_fea_info[i].set_y0(m_fea_info[i].y0()+m_shifty);
            m_fea_info[i].set_y1(m_fea_info[i].y1()+m_shifty);
        }
        return;
    }
    else
    {
        throw runtime_error("Feature Info is Empty");
    }
}

void FeaturesExtractor::setImage(const unsigned char* img_begin, int width, int height, int flags)
{
    m_integral_img_processor->setImage(img_begin,width,height,flags);
}

void FeaturesExtractor::compute(const vector<cv::Mat> &integral_channels,int shiftx, int shifty)
{
    m_features.clear();

    if(m_fea_info.empty())
    {
        throw runtime_error("features not ready");
    }

    for(int i=0;i<m_fea_info.size();i++)
    {
        const ICFDetector::LeafModel &fea = m_fea_info[i];
        if(fea.channel_index()>=integral_channels.size())
        {
            throw runtime_error("Feature Info Contains More Channels than readable Channels.");
        }

        const cv::Mat &timgs = integral_channels.at(fea.channel_index());

        int integral_f = timgs.at<int32_t>(fea.y1()+shifty,fea.x1()+shiftx)
                - (timgs.at<int32_t>(fea.y0()+shifty,fea.x1()+shiftx)+timgs.at<int32_t>(fea.y1()+shifty,fea.x0()+shiftx))
                + timgs.at<int32_t>(fea.y0()+shifty,fea.x0()+shiftx);

        m_features.push_back(integral_f);
    }
}

void FeaturesExtractor::compute(int shiftx, int shifty)
{
    m_features.clear();

    if(m_fea_info.empty())
    {
        throw runtime_error("features not ready");
    }

    m_integral_img_processor->compute();
    const vector<cv::Mat>& integral_channels = m_integral_img_processor->getIntegralImgs();
    for(int i=0;i<m_fea_info.size();i++)
    {
        const ICFDetector::LeafModel &fea = m_fea_info[i];
        if(fea.channel_index()>=integral_channels.size())
        {
            throw runtime_error("Feature Info Contains More Channels than readable Channels.");
        }

        const cv::Mat &timgs = integral_channels.at(fea.channel_index());

        // to debug
        if(false)
        {
            cout<<fea.x0()<<" "<<fea.x1()<<" "<<fea.y0()<<" "<<fea.y1()<<endl;

            int aa = timgs.at<int32_t>(fea.y1(),fea.x1());cout<<aa<<endl;
            aa = timgs.at<int32_t>(fea.y0(),fea.x1());cout<<aa<<endl;
            aa = timgs.at<int32_t>(fea.y1(),fea.x0());cout<<aa<<endl;
            aa = timgs.at<int32_t>(fea.y0(),fea.x0());cout<<aa<<endl;
            throw runtime_error("Debug Interrupt");
        }



        int integral_f = timgs.at<int32_t>(fea.y1()+shifty,fea.x1()+shiftx)
                - (timgs.at<int32_t>(fea.y0()+shifty,fea.x1()+shiftx)+timgs.at<int32_t>(fea.y1()+shifty,fea.x0()+shiftx))
                + timgs.at<int32_t>(fea.y0()+shifty,fea.x0()+shiftx);

        m_features.push_back(integral_f);
    }
}

const std::vector<double>& FeaturesExtractor::getFeatures() const
{
    return m_features;
}

