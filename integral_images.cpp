#include "integral_images.hpp"
#include "helpers.hpp"
#include <stdexcept>
//#include <cstdint>

using namespace std;

namespace{

/// coefficients values come from
/// www.doc.ic.ac.uk/~wl/papers/bf95.pdf
/// www.cse.yorku.ca/~kosta/CompVis_Notes/binomial_filters.pdf
/// http://en.wikipedia.org/wiki/Pascal%27s_triangle
std::vector<float> get_binomial_kernel_1d(const int binomial_filter_radius)
{
    const int radius = binomial_filter_radius;
    std::vector<float> coefficients;

    if(radius == 0)
    {
        coefficients.push_back(1.0f);
    }
    else if(radius == 1)
    {
        // 1 2 1
        const int sum = 4;
        coefficients.push_back(1.0f/sum);
        coefficients.push_back(2.0f/sum);
        coefficients.push_back(1.0f/sum);
    }
    else if(radius == 2)
    {
        // 1 4 6 4 1
        const int sum = 16;
        coefficients.push_back(1.0f/sum);
        coefficients.push_back(4.0f/sum);
        coefficients.push_back(6.0f/sum);
        coefficients.push_back(4.0f/sum);
        coefficients.push_back(1.0f/sum);
    }
    else
    {
        throw std::runtime_error("get_binomial_kernel_1d only accepts radius values 0, 1 && 2");
    }

    assert(coefficients.empty() == false);
    return coefficients;
}
}

IntegralImages::IntegralImages()
{
    const cv::Mat binomial_kernel_1d = cv::Mat(get_binomial_kernel_1d(1), true);
    m_pre_smooth_filter_p =
            cv::createSeparableLinearFilter(CV_8UC1, CV_8UC1, binomial_kernel_1d, binomial_kernel_1d,
                                            cv::Point(-1,-1),0,cv::BORDER_REPLICATE,cv::BORDER_REPLICATE);// equal to [1 2 1;2 4 2;1 2 1] or others

    const cv::Mat dx_kernel = (cv::Mat_<int8_t>(1, 3) << -1, 0, 1);
    m_dx_filter_p = cv::createLinearFilter(CV_8UC1, CV_16SC1, dx_kernel,
                                           cv::Point(-1,-1),0,cv::BORDER_REPLICATE,cv::BORDER_REPLICATE);

    const cv::Mat dy_kernel = (cv::Mat_<int8_t>(3, 1) << -1, 0, 1);
    m_dy_filter_p = cv::createLinearFilter(CV_8UC1, CV_16SC1, dy_kernel,
                                           cv::Point(-1,-1),0,cv::BORDER_REPLICATE,cv::BORDER_REPLICATE);

}

IntegralImages::~IntegralImages()
{

}

void IntegralImages::integrateMat(const cv::Mat &input, cv::Mat &output)
{
    output.create(input.rows+1,input.cols+1,CV_32SC1);

    // first row of the integral_channel, is set to zero
    for(int i=0;i<output.cols;i++)
    {
        output.at<int32_t>(0,i)=0;
    }

    bool debug=false;

    // we count rows in the integral_channel, they are shifted by one pixel from the image rows
    for(size_t row=1; row < output.rows; row++)
    {
        int32_t *integral_channel_previous_row = output.ptr<int32_t>(row-1);
        int32_t *integral_channel_row = output.ptr<int32_t>(row);
        const unsigned char* channel_row = input.ptr<unsigned char>(row-1);

        integral_channel_row[0] = 0;

        // integral_channel_row.size() == (channel_row.size() + 1) so everything is fine
        for(size_t col=0; col < input.cols; col++)
        {
            integral_channel_row[col+1] = integral_channel_previous_row[col+1]
                    - integral_channel_previous_row[col] +
                    channel_row[col] +
                    integral_channel_row[col];

            //debug
            if(false && col==2)
            {
                cout<<integral_channel_previous_row[col+1]<<" "<<integral_channel_previous_row[col]<<" "
                                                         <<(int)channel_row[col]<<" "<<(int)integral_channel_row[col];
                cout<<integral_channel_row[col+1]<<" ";
            }

        } // end of "for each column in the input channel"

        if(debug)
        {
            cv::Mat k = input(cv::Rect(0,0,5,5));
            cout<<k<<endl;
            debug=false;
        }
    }

    // to debug
    if(false)
    {
        for(int j=1;j<output.rows;j++)
        {
            for(int i=1;i<output.cols;i++)
            {
                bool check = (output.at<int32_t>(j,i) >= output.at<int32_t>(j,i-1)) && (output.at<int32_t>(j,i) >= output.at<int32_t>(j-1,i));
                if(!check)
                {
                    cout<<j<<" "<<i<<endl;
                    cout<<output.at<int32_t>(j,i)<<" "<<output.at<int32_t>(j-1,i)<<" "<<output.at<int32_t>(j,i-1)<<endl;
                    throw runtime_error("@.@");
                }
            }
        }
    }
    return;
}


void IntegralImages::setImage(const unsigned char* img_begin, int width, int height, int flags)
{
    unsigned char* img_data = const_cast<unsigned char*>(img_begin);
    switch(flags)
    {
    case 1:
        m_original_img = cv::Mat(height,width,CV_8UC1,img_data);
        break;
    default:
        throw runtime_error("Don't support flags not equal to 1 in FeaIntegralGradient");
    }

}

void IntegralImages::compute()
{
    static cv::Mat temp,dxmt,dymt;

    bool use_y_channel = (ICFHelp::getChannel()==1);
    bool use_gradient_channel = (ICFHelp::getChannel()==0);



    if(use_gradient_channel)
    {
        temp.create(m_original_img.rows,m_original_img.cols,CV_8UC1);
        dxmt.create(m_original_img.rows,m_original_img.cols,CV_16SC1);
        dymt.create(m_original_img.rows,m_original_img.cols,CV_16SC1);

        // Get Gradient Image
        m_pre_smooth_filter_p->apply(m_original_img,temp);
        //    temp = m_original_img; // without smooth, too noisy

        m_dx_filter_p->apply(temp,dxmt);
        m_dx_filter_p->apply(temp,dymt);
        m_gradient_img.create(temp.rows,temp.cols,CV_8UC1);
        unsigned char* pt = m_gradient_img.data;
        int16_t* px = reinterpret_cast<int16_t*>(dxmt.data);
        int16_t* py = reinterpret_cast<int16_t*>(dymt.data);
        int le = m_gradient_img.cols*m_gradient_img.rows;
        float norm_factor = 255.0/(sqrt(2.0)*255.0);
        for(int i=0;i<le;i++)
        {
            float temp = sqrt(1.0f*px[i]*px[i]+1.0f*py[i]*py[i])*norm_factor;
            if(temp>255||temp<0)
            {
                cout<<temp<<endl;
                throw runtime_error("MEME da");
            }
            pt[i] = static_cast<unsigned char>(round(temp));
        }

        //debug
        if(false)
        {
            static int i=0;
            stringstream a;
            a<<(i++)<<".png";
            cv::imwrite(a.str().c_str(),m_gradient_img);
        }


        // Do integrals
        cv::resize(m_gradient_img,m_gradient_img_shrinked,cv::Size(0,0),
                   1.0/ICFHelp::getShrinkFactor(),1.0/ICFHelp::getShrinkFactor());
        integrateMat(m_gradient_img_shrinked,m_integral_gradient_img);
        m_integral_imgs.clear();
        m_integral_imgs.push_back(m_integral_gradient_img);

        // to debug
        if(false)
        {
            cv::imwrite("smooth.png",temp);
            cv::imwrite("gradient.png",m_gradient_img);
        }
    }
    else if(use_y_channel)
    {
        cv::resize(m_original_img,m_gradient_img_shrinked,cv::Size(0,0),
                   1.0/ICFHelp::getShrinkFactor(),1.0/ICFHelp::getShrinkFactor(),cv::INTER_NEAREST);
        integrateMat(m_gradient_img_shrinked,m_integral_gradient_img);
        m_integral_imgs.clear();
        m_integral_imgs.push_back(m_integral_gradient_img);
    }
    else
    {
        throw runtime_error("Unknown Channel Catogary");
    }

    return;
}

// return value is in CV32SC1(int32_t) type
const vector<cv::Mat> &IntegralImages::getIntegralImgs() const
{
    return m_integral_imgs;
}

const cv::Mat& IntegralImages::getGradientImgs() const
{
    return m_gradient_img;
}

const cv::Mat& IntegralImages::getGradientImgs2() const
{
    static cv::Mat retv;
    retv.create(m_gradient_img.rows,m_gradient_img.cols,CV_8UC1);

    for(int j=0;j<retv.rows;j++)
    {
        for(int i=0;i<retv.cols;i++)
        {
            int temp = (m_integral_gradient_img.at<int>(j+1,i+1)-
                        m_integral_gradient_img.at<int>(j+1,i)-
                        m_integral_gradient_img.at<int>(j,i+1)+
                        m_integral_gradient_img.at<int>(j,i));

            if(temp<0||temp>255)
            {
                cout<<temp<<endl;
                throw runtime_error("How Come?");
            }
            retv.at<unsigned char>(j,i) = temp;
        }
    }
    return retv;
}

