import pandas as pd
from gaussian import gaussian_probability
class BayesianPredictor:
    def __init__(self,data):
        self.data=data
        self.diabetic=data[data["Outcome"]==1]
        self.normal=data[data["Outcome"]==0]
        self.prior_diabetic=len(self.diabetic)/len(data)
        self.prior_normal=len(self.normal)/len(data)
    def predict(self,sample):
        diabetic_prob=self.prior_diabetic
        normal_prob=self.prior_normal
        for feature,value in sample.items():
            mean1=self.diabetic[feature].mean()
            std1=self.diabetic[feature].std()
            mean0=self.normal[feature].mean()
            std0=self.normal[feature].std()
            diabetic_prob*=gaussian_probability(value,mean1,std1)
            normal_prob*=gaussian_probability(value,mean0,std0)
        total=diabetic_prob+normal_prob
        diabetic_prob/=total
        normal_prob/=total
        return diabetic_prob,normal_prob
