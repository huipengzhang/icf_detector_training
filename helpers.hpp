#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <vector>
#include "icf_detector.pb.h"

namespace cv {
class Mat;
}

namespace ICFHelp{

float getCascadeThreshold();
int getModelWidth(int mw=-1);
int getModelHeight(int mh=-1);
int getShrinkFactor(int sf=-1);
int getRandFeatureNum(int fn=-1);
int getTrainPosOffset();
int getTrainingFlag(int flag=-1);// 0: not training, 1: is training
int getChannel(int flag=-1);// 0: gradient, 1: y channel


void generate_random_leafs(std::vector<ICFDetector::LeafModel>& fea_info, int channel_num);
ICFDetector::StrongModel getRescaledStrongModel(const ICFDetector::StrongModel& inputmodel, float relative_scale);
ICFDetector::LeafModel getRescaledLeafModel(const ICFDetector::LeafModel& inputmodel, float relative_scale);

ICFDetector::StrongModel getScoreNormalize(const ICFDetector::StrongModel& inputmodel);


// test the weak model, return the final score
double weakModelJudge(const ICFDetector::WeakModel& mo, double sponse0, double sponseLeft, double sponseRight);
double weakModelJudge(const ICFDetector::WeakModel& mo, const std::vector<cv::Mat> &integral_channels, int shiftx, int shifty);

inline bool leafmodel_equal(const ICFDetector::LeafModel& a, const ICFDetector::LeafModel& b){
    return a.channel_index()==b.channel_index() && a.x0()==b.x0() && a.x1()==b.x1() && a.y0()==b.y0() && a.y1()==b.y1();
}

}

#endif
