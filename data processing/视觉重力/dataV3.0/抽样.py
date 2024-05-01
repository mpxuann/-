import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import random
import warnings
import os
import glob
random.seed()
warnings.filterwarnings('ignore')

dataPath = os.getcwd()
if not os.path.exists(os.path.join(dataPath,'figure')):
    os.makedirs(os.path.join(dataPath,'figure'))

pngFiles = glob.glob(os.path.join(dataPath,'figure','*.png'))
for file in pngFiles:
    os.remove(file)

def merge_csv_files(directory):
    all_files = os.listdir(directory)
    df_list = []
    for file in all_files:
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            df['ID_Index'] = os.path.splitext(file)[0]
            df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    return merged_df

def indTTest(DataFrame, Column1,Column2):
    result = scipy.stats.ttest_ind(DataFrame[Column1],DataFrame[Column2] )
    return result

def oneSamTTest(DataFrame, Column, mu):
    result = scipy.stats.ttest_1samp(DataFrame[Column],mu )
    return result

def selectParticipants(DataFrame, Column, Size, Replace):
    return DataFrame[DataFrame[Column].isin(np.random.choice(DataFrame[Column].unique(), size=Size, replace=Replace))]

def shootDirection(x):
    if x in [0,180]:
        return '水平'
    elif x in [45,135]:
        return '斜向上'
    elif x in [90,270]:
        return '竖直'
    elif x in [225,315]:
        return '斜向下'

def makeDict(NumOPar,SampleIndex, ExpIndex,ExpCondition,Error,pValue):
    dict = {
        '被试数量' : NumOPar,
        '样本序号' : SampleIndex,
        '实验序号' : [ExpIndex],
        '实验处理' : [ExpCondition],
        '误差' : [Error],
        'p值' : pValue
    }
    return dict

def hypothesisTest(DataFrame,DataSeries,SampleType,Conditions,NumOPar,SampleIndex,ExpNum):
    resultSingleExp = pd.DataFrame()
    if SampleType == 'Ratio':
        for multiIndex,i in DataSeries:
            Choose = i.sum()
            num = 10
            nub = len(i)*num
            pValue = scipy.stats.binom_test(Choose*num,nub,0.5,alternative='two-sided') #0.5双边检验
            Dict = makeDict(NumOPar,SampleIndex,ExpNum,multiIndex,Conditions,pValue)
            resultSingleCondition = pd.DataFrame(Dict)
            resultSingleExp = pd.concat([resultSingleExp,resultSingleCondition],axis=0)
        return resultSingleExp
    if SampleType == 'OneSample':
        for multiIndex, i in DataSeries:
            stats, pValue = oneSamTTest(DataFrame=i,Column=Conditions,mu=0)
            Dict = makeDict(NumOPar,SampleIndex,ExpNum,multiIndex,Conditions,pValue)
            resultSingleCondition = pd.DataFrame(Dict)
            resultSingleExp = pd.concat([resultSingleExp,resultSingleCondition],axis=0)
        return resultSingleExp
    
def processPerExp(DataFrame, ExpNum,NumOPar,SampleIndex):
    if ExpNum == 'Exp1':
        dfGrouped = DataFrame.groupby('cond')['ratio']
        resultSingleExp= pd.DataFrame()
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(DataFrame,dfGrouped,'Ratio','Nothing',NumOPar,SampleIndex,ExpNum)],axis=0)
        return resultSingleExp    
    if ExpNum == 'Exp2':
        dfGroupedAngle = DataFrame.groupby(['发射方向','速度'])[['水平误差','垂直误差']]
        dfGroupedDirection = DataFrame.groupby(['方向','速度'])[['水平误差','垂直误差']]
        resultSingleExp= pd.DataFrame()
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGroupedAngle,
                                                                    'OneSample','水平误差',NumOPar,SampleIndex,ExpNum)],axis=0)
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGroupedAngle,
                                                                    'OneSample','垂直误差',NumOPar,SampleIndex,ExpNum)],axis=0)   
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGroupedDirection,
                                                                    'OneSample','水平误差',NumOPar,SampleIndex,ExpNum)],axis=0)
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGroupedDirection,
                                                                    'OneSample','垂直误差',NumOPar,SampleIndex,ExpNum)],axis=0)
        return resultSingleExp    
    if ExpNum == 'Exp3':
        dfGrouped = DataFrame.groupby('moveCondition')[
            ['xIfNoGravityPredictError','yIfNoGravityPredictError','xIfGravityPredictError','yIfGravityPredictError']]
        resultSingleExp= pd.DataFrame()
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGrouped,
                                                                    'OneSample','xIfNoGravityPredictError',NumOPar,SampleIndex,ExpNum)],axis=0)
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGrouped,
                                                                    'OneSample','yIfNoGravityPredictError',NumOPar,SampleIndex,ExpNum)],axis=0)
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGrouped,
                                                                    'OneSample','xIfGravityPredictError',NumOPar,SampleIndex,ExpNum)],axis=0)
        resultSingleExp = pd.concat([resultSingleExp,hypothesisTest(pd.DataFrame(),dfGrouped,
                                                                    'OneSample','yIfGravityPredictError',NumOPar,SampleIndex,ExpNum)],axis=0)
        return resultSingleExp

def Drawfig(multiIndex,DataSeries,SamplingColumn,GroupColumn,TargetColumn,DataPath,picSaveName):
    plt.clf()
    Sampling = len(DataSeries[SamplingColumn].unique())
    grouped = DataSeries.groupby(GroupColumn)
    Median = grouped.median()
    Q1 = grouped.quantile(0.25)
    Q3 = grouped.quantile(0.75)
    plt.errorbar(Median.index, Median[TargetColumn], 
                 yerr=[Median[TargetColumn] - Q1[TargetColumn], Q3[TargetColumn] - Median[TargetColumn]], 
                 fmt='-o',capsize=3)
    plt.axhline(y=0.05,color = 'r', linestyle= '--')
    plt.xlabel('Subject Number')
    plt.ylabel('Median P Value')
    if multiIndex[0] == 'Exp1':
        plt.title(f'{multiIndex[0]}, {multiIndex[1]}, samplingTimes = {Sampling}',fontdict={'fontfamily':'Microsoft YaHei'})
    else:    
        plt.title(f'{multiIndex[0]}, {multiIndex[1]},{multiIndex[2]}, samplingTimes = {Sampling}',fontdict={'fontfamily':'Microsoft YaHei'})
    plt.grid(False)
    DataPath = os.path.join(DataPath,'figure')
    plt.savefig(os.path.join(DataPath,picSaveName))

df1 = pd.read_excel("exp1 V3.0.xlsx")
df2 = pd.read_excel("exp2 V3.0.xlsx")
df3 = pd.read_csv("exp3 V3.0.csv")
df2['方向'] = df2['发射方向'].apply(shootDirection)
# df1['response'] = df1['responses'].apply(lambda x : 1 if x=='R' else 0)

resultAllExp =pd.DataFrame()
for i in range(2,11): #抽取i个被试
    for j in range(1,21): #抽取i个被试时，取j次样本
        random.seed()
        df1Selected = selectParticipants(df1,'ID_Index',i,False)#抽取i个被试
        df2Selected = selectParticipants(df2,'ID_Index',i,False)
        df3Selected = selectParticipants(df3,'ID_Index',i,False)
        resultAllExp = pd.concat([resultAllExp,processPerExp(df1Selected,'Exp1',i,j)])#计算df1Selectedd各处理的显著性水平
        resultAllExp = pd.concat([resultAllExp,processPerExp(df2Selected,'Exp2',i,j)])
        resultAllExp = pd.concat([resultAllExp,processPerExp(df3Selected,'Exp3',i,j)])
resultAllExp.to_csv('抽样结果.csv',encoding='GB18030')

resultAllExp['实验序号'] = resultAllExp['实验序号'].astype(str)
resultAllExp['实验处理'] = resultAllExp['实验处理'].astype(str)
resultAllExp['误差'] = resultAllExp['误差'].astype(str)
dfGroup = resultAllExp.groupby(['实验序号','实验处理','误差'])[['被试数量','p值','样本序号']]
for multiIndex,DataSeries in dfGroup:
    if multiIndex[0] == 'Exp1':
        Drawfig(multiIndex,DataSeries,'样本序号','被试数量','p值',dataPath,f'{multiIndex[0]}-{multiIndex[1]}')
    if (multiIndex[0] == 'Exp2') or (multiIndex[0]=='Exp3'):
        Drawfig(multiIndex,DataSeries,'样本序号','被试数量','p值',dataPath,f'{multiIndex[0]}-{multiIndex[1]}-{multiIndex[2]}')

       
