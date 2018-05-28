#include "training_adaboost.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "integral_images.hpp"
#include "features_extractor.hpp"
#include "helpers.hpp"
#include <ctime>
#include <cmath>

using namespace std;

namespace
{
bool train_data_feature_compare(const ICFTrainDataLite& a, const ICFTrainDataLite& b)
{
    return a.data_feature<b.data_feature;
}

inline bool nodeLeftJudge(const ICFDetector::LeafModel& node, double data )
{
    return data<node.threshold();
}


struct TrainFeatureDescript
{
    TrainFeatureDescript():mfeai(-1),mfea_min_error_weight(1e100),mfea_scorethreshold(0),mfea_is_leftpos(true){}
    bool operator < (const TrainFeatureDescript& p)
    {
        return this->mfea_min_error_weight<p.mfea_min_error_weight;
    }

    int mfeai;
    double mfea_min_error_weight;
    double mfea_scorethreshold;
    bool mfea_is_leftpos;
};

}

TrainingAdaboost* TrainingAdaboost::getInstance()
{
    static TrainingAdaboost *g_ptr=NULL;
    if(g_ptr==NULL)
        g_ptr = new TrainingAdaboost();

    return g_ptr;
}


TrainingAdaboost::TrainingAdaboost()
{
    ICFHelp::generate_random_leafs(m_random_feas,1);
    m_pos_num=0;
    m_neg_num=0;
}

TrainingAdaboost::~TrainingAdaboost()
{

}

void TrainingAdaboost::readPositive(string pos_list_name)
{
    //NOTE - positive images should have edges of some pixels
    int real_offset = ICFHelp::getTrainPosOffset()/ICFHelp::getShrinkFactor();
    FeaturesExtractor fea_extract(m_random_feas,real_offset,real_offset);

    ifstream fin(pos_list_name.c_str());
    if(!fin.is_open())
    {
        cout<<"File Open Error: "<<pos_list_name<<endl;
        throw runtime_error("No Readable Positive List");
        return;
    }

    vector<string> fn_vec;
    string filename;
    fin>>filename;
    while(!fin.eof())
    {
        fn_vec.push_back(filename);
        fin>>filename;
    }
    if(fn_vec.empty())
    {
        cout<<"Empty Pos List: "<<pos_list_name<<endl;
        throw runtime_error("Empty Pos File");
    }

    int oi=m_training_data_vec.size();
    ICFTrainData template_traindata(ICFHelp::getRandFeatureNum());
    m_training_data_vec.resize(m_training_data_vec.size()+fn_vec.size(),template_traindata);
    for(int fi=0;fi<fn_vec.size();fi++)
    {
        cv::Mat pic = cv::imread(fn_vec[fi]);
        if(false)//debug
        {
            cv::namedWindow("kuku");
            cv::imshow("kuku",pic);
            cv::waitKey();
        }
        cv::cvtColor(pic,pic,CV_BGR2GRAY);
        fea_extract.setImage(pic.data,pic.cols,pic.rows,1);
        fea_extract.compute();
        m_training_data_vec[oi+fi].data_features = fea_extract.getFeatures();
        m_training_data_vec[oi+fi].data_label = 1;
        if(false){
            for(int j=0;j<10;j++)
            {
                cout<<m_training_data_vec[oi+fi].data_features[j]<<" ";
            }
            cout<<endl;
        }
    }

    m_pos_num += fn_vec.size();
}

void TrainingAdaboost::readBigNegative(std::string big_neg_list_name)
{
    using namespace cv;



    ifstream fin(big_neg_list_name.c_str());
    string fnname;
    std::getline(fin,fnname);
//    fin>>fnname;
    if(false)//debug
    {
        cout<<fnname<<endl;
        throw runtime_error("debug stop");
    }

    while(!fin.eof())
    {
        Mat readm = cv::imread(fnname);
        Mat graym;
        cvtColor(readm,graym,CV_BGR2GRAY);
        m_big_negative_names.push_back(fnname);
        m_big_negative_imgs.push_back(graym);

//        fin>>fnname;
        std::getline(fin,fnname);
    }
}

void TrainingAdaboost::cutRandomNegative(int max_random_pics)
{
    //NOTE - no edge is added to cut random negatives
    if(m_big_negative_imgs.empty())
        throw std::runtime_error("No big negative images are read in advance");


    FeaturesExtractor fea_extract(m_random_feas,0,0);

    if(false)
        srand(std::time(NULL));
    else
        srand(12);

    int oi = m_training_data_vec.size();
    ICFTrainData template_traindata(10000);
    m_training_data_vec.resize(m_training_data_vec.size()+max_random_pics,template_traindata);
    for(int i=0;i<max_random_pics;i++)
    {
        int j=i%m_big_negative_imgs.size();
        fea_extract.setImage(m_big_negative_imgs[j].data, m_big_negative_imgs[j].cols,m_big_negative_imgs[j].rows,1);

        int maxshiftx = (m_big_negative_imgs[j].cols-ICFHelp::getModelWidth())/ICFHelp::getShrinkFactor();
        int maxshifty = (m_big_negative_imgs[j].rows-ICFHelp::getModelHeight())/ICFHelp::getShrinkFactor();

        int shiftx = rand()%maxshiftx;
        int shifty = rand()%maxshifty;
        fea_extract.compute(shiftx,shifty);

        m_training_data_vec[oi+i].data_features = fea_extract.getFeatures();
        m_training_data_vec[oi+i].data_label = -1;

        // to debug
        if(false)
        {
            cout<<m_big_negative_names[j]<<"--"<<shiftx<<"--"<<shifty<<"--";
            for(int jj=0;jj<10;jj++)
            {
                cout<<m_training_data_vec[oi+i].data_features[jj]<<" ";
            }
            cout<<endl;
        }
    }

    m_neg_num += max_random_pics;
}



int TrainingAdaboost::addNegatives(const std::vector<ICFTrainData>& to_add_negs)
{
    m_training_data_vec.insert(m_training_data_vec.end(),to_add_negs.begin(),to_add_negs.end());
    m_neg_num+=to_add_negs.size();
    return to_add_negs.size();
}

void TrainingAdaboost::initNewTrain()
{
    cout<<"balance training dataset and set sum_score"<<endl;
    for(int i=0;i<m_training_data_vec.size();i++)
    {
        m_training_data_vec[i].sum_score = 0;

        if(m_training_data_vec[i].data_label<0)//neg
            m_training_data_vec[i].weight = 0.5/m_neg_num;
        else
            m_training_data_vec[i].weight = 0.5/m_pos_num;
    }
}

template< typename IteratorType>
ICFDetector::LeafModel TrainingAdaboost::trainLeafModel(IteratorType train_data_first,
                                                        IteratorType train_data_last,
                                                        const std::vector<ICFDetector::LeafModel>& features,
                                                        int& feai,
                                                        double& min_error_weight)
{
    feai=-1;
    min_error_weight=1e100;

    ICFDetector::LeafModel return_node;
    if(features.empty())
        throw runtime_error("Features Empty in Training Leaf Model");

    if(train_data_first == train_data_last)
    {
        cout<<"No Data Need To Be Trained!"<<endl;
        return_node = features.at(0);
        return_node.set_score(1);
        return_node.set_threshold(1e10);
        feai=0;
        min_error_weight=0;
        cout<<"Hack it!"<<endl;
        return return_node;
    }

    // get pos/neg weight sum
    double pos_weight_sum=0,neg_weight_sum=0;
    for(IteratorType j=train_data_first;j<train_data_last;j++)
    {
        if(j->data_label<0) // negative
        {
            neg_weight_sum += j->weight;
        }
        else // positive
        {
            pos_weight_sum += j->weight;
        }
    }

    vector<TrainFeatureDescript> train_re(features.size());

#pragma omp parallel for
    for(int fi=0;fi<features.size();fi++) // for each features
    {
        std::vector<ICFTrainDataLite> train_data_lite(train_data_last-train_data_first);
        for(int i=0;i<train_data_lite.size();i++)
        {
            train_data_lite[i]=ICFTrainDataLite(*(train_data_first+i),fi);
        }
        sort(train_data_lite.begin(),train_data_lite.end(),train_data_feature_compare);


        // get min error and proper score
        double
                //error if examples in the left are positive
                positivesLeft_LeftError = 0,
                positivesLeft_RightError = pos_weight_sum,

                //error if examples in the left are negative
                negativesLeft_LeftError = 0,
                negativesLeft_RightError = neg_weight_sum;

        double score_threshold=train_data_lite[0].data_feature*0.98,minerror=pos_weight_sum;
        bool posLeft=true;
        for (int k = 0; k < train_data_lite.size(); ++k) // loop within data
        {
            if(train_data_lite[k].data_label<0)//neg
            {
                positivesLeft_LeftError += train_data_lite[k].weight;
                negativesLeft_RightError -= train_data_lite[k].weight;
            }
            else //pos
            {
                positivesLeft_RightError -= train_data_lite[k].weight;
                negativesLeft_LeftError += train_data_lite[k].weight;
            }


            double tempError = 0;
            bool tempPosLeft = true;
            if(positivesLeft_LeftError + positivesLeft_RightError < negativesLeft_LeftError + negativesLeft_RightError)
            {
                tempError = positivesLeft_LeftError + positivesLeft_RightError;
                tempPosLeft = true;
            }
            else
            {
                tempError = negativesLeft_LeftError + negativesLeft_RightError;
                tempPosLeft = false;
            }


            // we keep the min error
            if (tempError < minerror)
            {
                minerror = tempError;
                posLeft = tempPosLeft;
                if(k < train_data_lite.size()-1)
                    score_threshold = (train_data_lite[k].data_feature+train_data_lite[k+1].data_feature)/2.0;
                else
                    score_threshold = train_data_lite[k].data_feature*1.01;

            }
        }// end of training data loop


        train_re[fi].mfeai = fi;
        train_re[fi].mfea_min_error_weight = minerror;
        train_re[fi].mfea_scorethreshold = score_threshold;
        train_re[fi].mfea_is_leftpos = posLeft;

    }// end of feature loop


    vector<TrainFeatureDescript>::iterator miniter = min_element(train_re.begin(),train_re.end());

    min_error_weight = miniter->mfea_min_error_weight;
    feai = miniter->mfeai;

    return_node = features.at(miniter->mfeai);
    return_node.set_threshold(round(miniter->mfea_scorethreshold));
    return_node.set_score(miniter->mfea_is_leftpos?1:-1);
    return return_node;
}

ICFDetector::WeakModel TrainingAdaboost::trainWeakModel(double &errorrate)
{
    if(m_training_data_vec.empty())
        throw std::runtime_error("Empty Training Data to train");


    ICFDetector::WeakModel retv;// weak model is a 2-level tree




//    cout<<"l0 train"<<endl;
    int feai;
    double dummy_weight;
    ICFDetector::LeafModel* tl0node = retv.mutable_l0_node();
    *tl0node = trainLeafModel(m_training_data_vec.begin(),m_training_data_vec.end(),
                              m_random_feas,feai,dummy_weight);


//    cout<<"prepare for level 1 train"<<endl;
    int start=0,end=m_training_data_vec.size()-1;
    while(true)
    {
        while(start<m_training_data_vec.size() && nodeLeftJudge(*tl0node,m_training_data_vec[start].data_features[feai]))
            start++;
        while(end>=0 && !nodeLeftJudge(*tl0node,m_training_data_vec[end].data_features[feai]))
            end--;

        if(start<end)
            swap(m_training_data_vec[start],m_training_data_vec[end]);
        else
            break;
    }
    int fea1left,fea1right;
    double errleft,errright;
//    cout<<"l1 train left"<<endl;
    ICFDetector::LeafModel* tl1node_small = retv.mutable_l1_node_small();
    *tl1node_small = trainLeafModel(m_training_data_vec.begin(),m_training_data_vec.begin()+start,
                                    m_random_feas,
                                    fea1left,errleft);

//    cout<<"l1 train right"<<endl;
    ICFDetector::LeafModel* tl1node_large = retv.mutable_l1_node_large();
    *tl1node_large = trainLeafModel(m_training_data_vec.begin()+start,m_training_data_vec.end(),
                                    m_random_feas,
                                    fea1right,errright);

//    cout<<"update weights"<<endl;
    double weightsum=0;
    for(int i=0;i<m_training_data_vec.size();i++)
    {
        weightsum+=m_training_data_vec[i].weight;
    }
    errorrate = (errleft+errright)/weightsum;

    double alpha_t = 14;
    if(errorrate>1e-12)
    {
        double p = (1-errorrate)/errorrate;
        alpha_t = 0.5*log(p);
    }
    retv.mutable_l1_node_small()->set_score(alpha_t*retv.l1_node_small().score());
    retv.mutable_l1_node_large()->set_score(alpha_t*retv.l1_node_large().score());

    double new_weightsum=0,totalerrorrate=0;
    for(int i=0;i<m_training_data_vec.size();i++)
    {
        double score = ICFHelp::weakModelJudge(retv,
                                               m_training_data_vec[i].data_features[feai],
                                               m_training_data_vec[i].data_features[fea1left],
                                               m_training_data_vec[i].data_features[fea1right]);
        m_training_data_vec[i].sum_score+=score;
        totalerrorrate+=(m_training_data_vec[i].sum_score*m_training_data_vec[i].data_label>0)?0:1;

        if( (m_training_data_vec[i].data_label<0 && score<0)
                ||(m_training_data_vec[i].data_label>0 && score>0)) // is correctly classified
        {
            m_training_data_vec[i].weight*=exp(-alpha_t);
        }
        else
        {
            m_training_data_vec[i].weight*=exp(alpha_t);
        }

        new_weightsum+=m_training_data_vec[i].weight;
    }

    for(int i=0;i<m_training_data_vec.size();i++)
    {
        m_training_data_vec[i].weight/=new_weightsum;
    }


    errorrate = totalerrorrate/m_training_data_vec.size();

    return retv;
}

int TrainingAdaboost::getPosNum()
{
    return m_pos_num;
}

int TrainingAdaboost::getNegNum()
{
    return m_neg_num;
}

