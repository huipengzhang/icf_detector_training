#include "helpers.hpp"
#include <boost/cstdint.hpp>
#include <boost/random.hpp>
#include <opencv2/opencv.hpp>
#include <cstdio>
using namespace std;


namespace ICFHelp {

float getCascadeThreshold()
{
    return -0.2;
}
int getModelWidth(int wd)
{
    static int model_wid = 24;

    if(wd>0)
        model_wid = wd;

    return model_wid;
}
int getModelHeight(int hei)
{
    static int model_hei = 24;

    if(hei>0)
        model_hei = hei;

    return model_hei;
}
int getShrinkFactor(int sf)
{
    static int shrink_factor = 2;

    if(sf>0)
        shrink_factor = sf;

    return shrink_factor;
}
int getRandFeatureNum(int tfnum)
{
    static int fnum = 3000;
    fnum = tfnum<0?fnum:tfnum;
    return fnum;
}
int getTrainPosOffset()
{
    return 0;
}
int getTrainingFlag(int flag)
{
    static int gflag=0;
    if(flag>=0)
        gflag = flag;

    return gflag;
}
int getChannel(int flag)
{
    static int g_channel = 0;
    if(flag>=0)
        g_channel = flag;

    return g_channel;
}





double weakModelJudge(const ICFDetector::WeakModel& mo, double sponse0, double sponseLeft, double sponseRight)
{
    if(sponse0<mo.l0_node().threshold())
    {
        if(sponseLeft<mo.l1_node_small().threshold())
        {
            return mo.l1_node_small().score();
        }
        else
        {
            return -mo.l1_node_small().score();
        }
    }
    else
    {
        if(sponseRight<mo.l1_node_large().threshold())
        {
            return mo.l1_node_large().score();
        }
        else
        {
            return -mo.l1_node_large().score();
        }
    }
}

double weakModelJudge(const ICFDetector::WeakModel& mo, const std::vector<cv::Mat> &integral_channels, int shiftx, int shifty)
{
    // level 1
    int c0 = mo.l0_node().channel_index();
    const cv::Mat &timgs0 = integral_channels.at(c0);
    int response0 = timgs0.at<int32_t>(mo.l0_node().y1()+shifty,mo.l0_node().x1()+shiftx)
            - (timgs0.at<int32_t>(mo.l0_node().y0()+shifty,mo.l0_node().x1()+shiftx)
               +timgs0.at<int32_t>(mo.l0_node().y1()+shifty,mo.l0_node().x0()+shiftx))
            + timgs0.at<int32_t>(mo.l0_node().y0()+shifty,mo.l0_node().x0()+shiftx);
    ICFDetector::LeafModel l1node;
    if(response0<mo.l0_node().threshold())
    {
        l1node = mo.l1_node_small();
    }
    else
    {
        l1node = mo.l1_node_large();
    }

    // level 2
    int c1 = l1node.channel_index();
    const cv::Mat &timgs1 = integral_channels.at(c1);
    int response1 = timgs1.at<int32_t>(l1node.y1()+shifty,l1node.x1()+shiftx)
            - (timgs1.at<int32_t>(l1node.y0()+shifty,l1node.x1()+shiftx)
               +timgs1.at<int32_t>(l1node.y1()+shifty,l1node.x0()+shiftx))
            + timgs1.at<int32_t>(l1node.y0()+shifty,l1node.x0()+shiftx);

    if(response1<l1node.threshold())
        return l1node.score();
    else
        return -l1node.score();
}

void generate_random_leafs(std::vector<ICFDetector::LeafModel>& fea_info, int numChannels)
{
    // NOTE: number of random sample features
    size_t total_num_of_features = getRandFeatureNum();

    const int shrinking_factor = getShrinkFactor(),
            modelWidth = getModelWidth(),
            modelHeight = getModelHeight();

    const int
            minWidth = max(4,modelWidth/8)/shrinking_factor,
            minHeight = max(4,modelHeight/8)/shrinking_factor,
            maxWidth = modelWidth/shrinking_factor,
            maxHeight = modelHeight/shrinking_factor;


    boost::uint32_t random_seed = 12;//std::time(NULL);
    boost::mt19937 random_generator(random_seed);
    typedef boost::variate_generator<boost::mt19937&, boost::uniform_int<> > uniform_generator_t;

    // the distribution boundaries are inclusive
    cout<<maxWidth<<" "<<maxHeight<<endl;
    boost::uniform_int<>
            x_distribution(0, (maxWidth - 1) - minWidth),
            y_distribution(0, (maxHeight -1) - minHeight),
            channel_distribution(0, numChannels - 1),
            width_distribution(minWidth, maxWidth - 1),
            height_distribution(minHeight, maxHeight - 1);

    if((x_distribution.max() <= 0) ||
            (y_distribution.max() <= 0) ||
            (width_distribution.max() <= 0) ||
            (height_distribution.max() <= 0))
    {
        printf("shrinked model (width, height) == (%i, %i)\n", maxWidth, maxHeight);
        printf("min feature size (after shrinking) (width, height) == (%i, %i)\n", minWidth, minHeight);
        throw invalid_argument("It seems that minFeatWidth or minFeatHeight is bigger than the model size after shrinking");
    }

    uniform_generator_t
            x_generator(random_generator, x_distribution),
            y_generator(random_generator, y_distribution),
            channel_generator(random_generator, channel_distribution),
            width_generator(random_generator, width_distribution),
            height_generator(random_generator, height_distribution);

    fea_info.clear();
    fea_info.reserve(total_num_of_features);

    int rejectionsInARow = 0, repetitionsCounter = 0;
    const int maxRejectionsInARow = 1000; // how many continuous rejection do we accept ?
    ICFDetector::LeafModel fea_leaf;
    fea_leaf.set_score(0);
    fea_leaf.set_threshold(0);

    int tc,tx,ty,tw,th;
    while(fea_info.size() < total_num_of_features)
    {
        tc = channel_generator();
        tx = x_generator();
        ty = y_generator();
        tw = width_generator();
        th = height_generator();
        if(((tx + tw) < maxWidth) && ((ty + th) < maxHeight))
        {
            fea_leaf.set_channel_index(tc);
            fea_leaf.set_x0(tx);
            fea_leaf.set_y0(ty);
            fea_leaf.set_x1(tx+tw);
            fea_leaf.set_y1(ty+th);

            // we check if the feature already exists in the set or not
            bool already_set = false;
            for(int i=0;i<fea_info.size();i++)
            {
                if( leafmodel_equal(fea_info[i],fea_leaf))
                {
                    already_set = true;
                    break;
                }
            }

            if(already_set)
            {
                rejectionsInARow ++;
                repetitionsCounter ++;
                if(rejectionsInARow > maxRejectionsInARow)
                {
                    printf("once featuresPool reached size %zi, failed to find a new feature after %i attempts\n",
                           fea_info.size(), maxRejectionsInARow);
                    throw std::runtime_error("Failed to generate the requested features pool, is featuresPoolSize too big?");
                }
                continue;
            }
            else
            {
                rejectionsInARow = 0;
                fea_info.push_back(fea_leaf);
            }
        } // end of "if the random feature has proper size"
    } // end of "while not enough features computed"

    printf("When sampling %zi features, randomly found (and rejected) %i repetitions\n",
           total_num_of_features, repetitionsCounter);

}


/// Helper method that gives the crucial information for the FPDW implementation
/// these numbers are obtained via
/// doppia/src/test/objects_detection/test_objects_detection + plot_channel_statistics.py
/// (this method is not speed critical)
///
///
///
///
/// Rescale the model to relative_scale, equal to calculate the scale when image shrink to 1/relative_scale
float getChannelRescaleFactor(float relative_scale,int channel_index)
{

    //Hint: exp(lambda*ln(r)/ln(2)) == pow(r,lambda/ln(2))

    float channel_scaling = 1, up_a = 1, down_a = 1, up_b = 2, down_b = 2;


    if(relative_scale == 1)
    { // when no rescaling there is no scaling factor
        return 1.0f;
    }

    relative_scale = 1.0/relative_scale;
    int channel = ICFHelp::getChannel();
    const bool use_p_dollar_estimates = true;
    if(use_p_dollar_estimates)
    {
        const float lambda = 1.099, a = 0.89;

        if(channel == 0)
        { // Gradient histograms && gradient magnitude
            down_a = a; down_b = lambda / log(2.0);
            // upscaling case is roughly a linear growth
            // these are the ideal values
            up_a = 1; up_b = 1;

        }
        else if(channel == 1)
        {
            // Y channels, quadratic growth
            // these are the ideal values
            down_a = 1; down_b = 2;
            up_a = 1; up_b = 2;
        }
        else
        {
            throw std::runtime_error("get_channel_scaling_factor use_p_dollar_estimates called with "
                                     "an unknown integral channel index");
        }

    }
    else
    {
        throw std::runtime_error("no estimate was selected for get_channel_scaling_factor");
    }


    {
        float a=1, b=2;
        if(relative_scale >= 1)
        { // upscaling case
            a = up_a;
            b = up_b;
        }
        else
        { // size_scaling < 1, downscaling case
            a = down_a;
            b = down_b;
        }

        channel_scaling = a*pow(relative_scale, b);

        const bool check_scaling = true;
        if(check_scaling)
        {
            if(relative_scale >= 1)
            { // upscaling
                if(channel_scaling < 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
            else
            { // downscaling
                if(channel_scaling > 1)
                {
                    throw std::runtime_error("get_channel_scaling_factor upscaling parameters are plain wrong");
                }
            }
        } // end of check_scaling
    }


    channel_scaling = 1.0/channel_scaling;

    return channel_scaling;
}


ICFDetector::LeafModel getRescaledLeafModel(const ICFDetector::LeafModel& inputmodel, float relative_scale)
{
    ICFDetector::LeafModel retv = inputmodel;
    const float channel_scaling_factor = getChannelRescaleFactor(relative_scale,inputmodel.channel_index());
    float ori_area = (retv.x1()-retv.x0())*(retv.y1()-retv.y0());
    retv.set_x0(round(retv.x0()*relative_scale));
    retv.set_y0(round(retv.y0()*relative_scale));
    retv.set_x1(round(retv.x1()*relative_scale));
    retv.set_y1(round(retv.y1()*relative_scale));
    float new_area = (retv.x1()-retv.x0())*(retv.y1()-retv.y0());

    float area_approximation_scaling_factor = 1;// to compensate the impact of round(float)-->int
    if((new_area > 0) && (ori_area > 0))
    {
        const float expected_new_area = ori_area*relative_scale*relative_scale;
        area_approximation_scaling_factor = expected_new_area / new_area;
    }

    retv.set_threshold(retv.threshold() / area_approximation_scaling_factor * channel_scaling_factor); // FIXME original code is wrong??

    if(false)//debug
    {
        printf("relative_scale %.3f -> channel_scaling_factor %.3f\n", relative_scale, channel_scaling_factor);
    }

    return retv;
}

ICFDetector::StrongModel getRescaledStrongModel(const ICFDetector::StrongModel& inputmodel, float relative_scale)
{
    ICFDetector::StrongModel retv = inputmodel;
    int modelsize = retv.weak_models_size();
    for(int i=0;i<modelsize;i++)
    {
        ICFDetector::WeakModel* wp = retv.mutable_weak_models(i);
        ICFDetector::LeafModel* temp;
        temp = wp->mutable_l0_node();
        *temp = getRescaledLeafModel(*temp,relative_scale);
        temp = wp->mutable_l1_node_small();
        *temp = getRescaledLeafModel(*temp,relative_scale);
        temp = wp->mutable_l1_node_large();
        *temp = getRescaledLeafModel(*temp,relative_scale);
    }

    return retv;
}


ICFDetector::StrongModel getScoreNormalize(const ICFDetector::StrongModel& inputmodel)
{
    ICFDetector::StrongModel retv = inputmodel;

    double scoresum=0;
    for(int i=0;i<inputmodel.weak_models_size();i++)
    {
        scoresum+=abs(inputmodel.weak_models(i).l1_node_small().score());
    }

    for(int i=0;i<inputmodel.weak_models_size();i++)
    {
        retv.mutable_weak_models(i)->mutable_l1_node_small()->set_score(
                    inputmodel.weak_models(i).l1_node_small().score()/scoresum);
        retv.mutable_weak_models(i)->mutable_l1_node_large()->set_score(
                    inputmodel.weak_models(i).l1_node_large().score()/scoresum);
    }


    if(false)//debug
    {
        double scoresum1=0;
        for(int i=0;i<retv.weak_models_size();i++)
        {
            scoresum1+=abs(retv.weak_models(i).l1_node_small().score());

        }
        cout<<scoresum1<<endl;
        throw std::runtime_error("Debug Rescore Stop");
    }

    return retv;
}













}

