#ifndef INTEGRAL_FEATURES_GRADIENT_HPP
#define INTEGRAL_FEATURES_GRADIENT_HPP

#include "icf_detector.pb.h"
#include "feature_extract.hpp"
#include <boost/shared_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


class IntegralImages;

class FeaturesExtractor
{
public:
    FeaturesExtractor(const std::vector<ICFDetector::LeafModel>& leafs, int shiftx=0, int shifty=0);
    virtual ~FeaturesExtractor();
    virtual const std::vector<double>& getFeatures() const ;
    virtual void setImage(const unsigned char* img_begin, int width, int height, int flags);// flags means channels
    virtual void compute(int shiftx=0,int shifty=0); // compute use pre-set image
    virtual void compute(const std::vector<cv::Mat> &integral_channels,int shiftx=0, int shifty=0);

private:
    virtual void initFeatureInfo(const std::vector<ICFDetector::LeafModel>& leafs);

private:
    int m_shiftx,m_shifty;
    boost::shared_ptr<IntegralImages> m_integral_img_processor;
    std::vector<ICFDetector::LeafModel> m_fea_info;
    std::vector<double> m_features;
};


#endif
