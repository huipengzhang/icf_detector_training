#ifndef TRAINING_ADABOOST_HPP
#define TRAINING_ADABOOST_HPP

#include <vector>
#include <climits>
#include "icf_detector.pb.h"
#include "data_types.hpp"

namespace cv
{
    class Mat;
}

struct ICFTrainData
{
    ICFTrainData(int fea_dims):sum_score(0),data_label(INT_MIN),weight(1),data_features(fea_dims){}

    double sum_score; // will be set automately
    int data_label; // negs: <0; pos: >=0
    double weight; // will be balanced automately
    std::vector<double> data_features; // must be pre-caculated
};

struct ICFTrainDataLite
{
    ICFTrainDataLite()
        :sum_score(0),data_label(INT_MIN),weight(1),data_feature(-1){}

    ICFTrainDataLite(const ICFTrainData& parent,int fi)
        :sum_score(parent.sum_score),
          data_label(parent.data_label),
          weight(parent.weight),
          data_feature(parent.data_features[fi]){}

    double sum_score;
    int data_label;
    double weight;
    double data_feature;
};

class TrainingAdaboost
{
public:
    template<typename IterType>
    static ICFDetector::LeafModel trainLeafModel(IterType input_data_first, IterType input_data_last,
                                                 const std::vector<ICFDetector::LeafModel>& feas,
                                                 int& feai,
                                                 double& errorweight);
    static TrainingAdaboost* getInstance();

    virtual ~TrainingAdaboost();

    void readPositive(std::string pos_list_name);
    void readBigNegative(std::string big_neg_list_name);
    void cutRandomNegative(int max_pic);
    int addNegatives(const std::vector<ICFTrainData>& to_add_negs);

    void initNewTrain();
    ICFDetector::WeakModel trainWeakModel(double& errorrate);

    int getPosNum();
    int getNegNum();

private:
    TrainingAdaboost();


public:
    std::vector< ICFDetector::LeafModel > m_random_feas;

private:
    int m_pos_num;
    int m_neg_num;

    std::vector<std::string> m_big_negative_names;
    std::vector<cv::Mat> m_big_negative_imgs;
    std::vector< ICFTrainData > m_training_data_vec;
};



#endif
