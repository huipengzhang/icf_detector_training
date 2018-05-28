#ifndef INTEGRAL_IMAGE_HPP
#define INTEGRAL_IMAGE_HPP

#include <opencv2/opencv.hpp>
#include <vector>



class IntegralImages
{
public:
    //NOTE only do integral images but not compute gradients here
    static void integrateMat(const cv::Mat& input, cv::Mat& output);

    IntegralImages();
    virtual ~IntegralImages();

    virtual void setImage(const unsigned char* img_begin, int width, int height, int flags);// flags means channels    
    virtual void compute();// compute integrals of channel images, shrinked with shrink_factor
    const std::vector<cv::Mat> &getIntegralImgs() const; // get integral channels stored in 32SC1

    // to debug
    const cv::Mat& getGradientImgs() const;
    const cv::Mat& getGradientImgs2() const;

private:

    cv::Ptr<cv::FilterEngine> m_pre_smooth_filter_p;
    cv::Ptr<cv::FilterEngine> m_dx_filter_p;
    cv::Ptr<cv::FilterEngine> m_dy_filter_p;

    cv::Mat m_original_img;
    cv::Mat m_gradient_img;
    cv::Mat m_gradient_img_shrinked;
    cv::Mat m_integral_gradient_img;

    std::vector<cv::Mat> m_integral_imgs;

};


#endif
