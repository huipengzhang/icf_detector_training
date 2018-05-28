#ifndef ICF_MULTI_DETECTOR_HPP
#define ICF_MULTI_DETECTOR_HPP


#include "icf_detector.pb.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "data_types.hpp"
#include "integral_images.hpp"


class ICFMultiDetector
{
public:
    ICFMultiDetector(std::string modelname,float minscale,float maxscale,int scalenum,
                     float scale1stridex,float scale1stridey, float threshold, float cascade_threshold);
    virtual ~ICFMultiDetector();

    void setImage(const cv::Mat& img);
    void setImage(const cv::Mat& img, const cv::Mat& mask);
    void compute();
    const std::vector<DetectResult>& getResults() const;

protected:
    void updateSearchParam(int wid, int height);
    void nonMaximumSuppression();
    ICFDetector::StrongModel getProperModel(float log_scale);

protected:
    // other
    IntegralImages m_integral_impl;

    // I/O variables
    cv::Mat m_original_img;// storing gray image
    cv::Mat m_shrinked_mask;// storing gray image representing mask
    bool m_use_mask;
    std::vector<DetectResult> m_detections;



    // searching used variables
    float m_min_scale;
    float m_max_scale;
    int m_scale_num;

    float m_scale1_stride_x;
    float m_scale1_stride_y;
    float m_threshold;
    float m_cascade_threshold;
    std::vector<SearchParam> m_search_params;// parameters used in searching, in shrinked integral images

    ICFDetector::MultiScaleStrongModel m_multi_scale_model;

};


#endif
