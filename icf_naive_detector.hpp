#ifndef ICF_NAIVE_DETECTOR_HPP
#define ICF_NAIVE_DETECTOR_HPP


#include "icf_detector.pb.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include "data_types.hpp"
#include "integral_images.hpp"
#include "training_adaboost.hpp"


class ICFNaiveDetector
{
public:
    ICFNaiveDetector(std::string modelname,float minscale,float maxscale,int scalenum,
                     float scale1stridex,float scale1stridey, float threshold);
    virtual ~ICFNaiveDetector();

    void setImage(const cv::Mat& img);
    std::vector<ICFTrainData> compute();
    const std::vector<DetectResult>& getResults() const;

protected:
    void updateSearchParam(int wid, int height);
    void nonMaximumSuppression();
    std::vector<DetectResult> nonMaximumSuppressionUnrescaled(const std::vector<DetectResult>& dets);

protected:
    // other
    IntegralImages m_integral_impl;

    // I/O variables
    cv::Mat m_original_img;// storing gray image
    std::vector<DetectResult> m_detections;

    // searching used variables
    float m_min_scale;
    float m_max_scale;
    int m_scale_num;

    float m_scale1_stride_x;
    float m_scale1_stride_y;
    float m_threshold;
    std::vector<SearchParam> m_search_params;

    ICFDetector::StrongModel m_scale1_model;

};


#endif
