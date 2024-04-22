import os
import sys
import glob

DIRNAME = os.path.dirname(__file__)
import math
import glob

dirName = os.path.dirname(__file__)
sys.path.append(os.path.join(dirName))
sys.path.append(os.path.join(dirName, '..'))
sys.path.append(os.path.join(dirName, '..', '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import researchpy as rp
from collections import OrderedDict


def readParametersFromDf(oneConditionDf):
    indexLevelNames = oneConditionDf.index.names
    parameters = {levelName: oneConditionDf.index.get_level_values(levelName)[0] for levelName in indexLevelNames}
    return parameters

def transformToAnnotation(pValue):
    if pValue < 0.05:
        return '*'
    elif pValue < 0.01:
        return '**'
    elif pValue < 0.001:
        return '***'
    else:
        return ''

def drawPerformanceLine(dataSeries, axForDraw, measure, color, nubOfSubj):
    order = dataSeries.index.get_level_values(2).unique().tolist()
    dataSeriesMean = dataSeries.groupby(level=2).mean()
    dataSeriesMean = dataSeriesMean.reindex(order)
    dataSeriesSte = dataSeries.groupby(level=2).std() / math.sqrt(nubOfSubj - 1)
    dataSeriesSte = dataSeriesSte.reindex(order)
    dataSeriesMean.plot(kind='bar', ax=axForDraw, y='mean', logx=False, label=measure, yerr=dataSeriesSte, capsize=4, color=color)
    axForDraw.set_ylim(0.0, 1.0)
    axForDraw.set_xlim(-0.5, 2.5)
    axForDraw.set_xticklabels(order, rotation=0, fontsize=16)
# 
def drawingResultantSubgraph(drawDictLevel0, numRows, numColumns, measure, color, nubOfSubj, drawPerformanceLine, picSaveName, dataPath):
    fig = plt.figure(figsize=(16, 10))
    plotCounter = 1
    for columnValue, dataSeriesLevel0 in drawDictLevel0.items():
        conditionOrderLevel1 = dataSeriesLevel0.index.get_level_values(1).unique().tolist()
        drawDictLevel1 = {condition: dataSeriesLevel0.groupby(level=1).get_group(condition) for condition in conditionOrderLevel1}
        for rowValue, dataSeriesLevel1 in drawDictLevel1.items():
            axForDraw = fig.add_subplot(numColumns, numRows, plotCounter)
            if plotCounter % numRows == 1:
                axForDraw.set_ylabel(columnValue if columnValue != 'padding' else '', fontsize=16)
            if plotCounter <= numRows:
                axForDraw.set_title(rowValue if rowValue != 'padding' else '', fontsize=16)
            axForDraw.tick_params(axis='both', labelsize=16)
            drawPerformanceLine(dataSeriesLevel1, axForDraw, measure, color, nubOfSubj)
            plotCounter += 1
            fig.subplots_adjust(top=0.9)
            plt.xlabel('')
            plt.legend(loc='upper right', fontsize=16)
    plt.suptitle('nubOfSubj={}'.format(nubOfSubj), x=0.51, fontsize=18)
    plt.savefig(os.path.join(dataPath,picSaveName))
    fig.clf()


def independentSampleTtest(dataIndexByConditionAndSample, variablesCompared):
    samplesGroupedByCondition = dataIndexByConditionAndSample.groupby(level=0)
    if not samplesGroupedByCondition.ngroups == 2:
        return -1, 'error'
    # pandas multi-indexing level 0 = isBoundary
    result = rp.ttest(group1=samplesGroupedByCondition.get_group(variablesCompared[0]),
                      group2=samplesGroupedByCondition.get_group(variablesCompared[1]),
                      paired=False)
    # print(result)
    pValue = result[1]['results'][3]
    return pValue, result


class AnalyzeDiffTrainConMasterWolf:
    def __init__(self, independentSampleTtest):
        self.independentSampleTtest = independentSampleTtest

    def __call__(self, groupedByData, variablesCompared):
        groupedByMasterWolf = groupedByData.groupby(level=1).get_group('masterWolf')
        pValue, result = self.independentSampleTtest(groupedByMasterWolf, variablesCompared)
        return result

# class AnalyzeDiffLineCon:
#     def __init__(self, independentSampleTtest):
#         self.independentSampleTtest = independentSampleTtest
#
#     def __call__(self, groupedByData, variablesCompared):
#         name = 'allianceTimes'
#         groupedByMasterWolf = groupedByData.groupby(level=1).get_group('masterWolf')
#         pValue, result = self.independentSampleTtest(groupedByMasterWolf, variablesCompared)
#         return result

class DrawDiffConResults:
    def __init__(self, drawPerformanceLine, drawingResultantSubgraph, dataPath, nubOfSubj, drawEachSub):
        self.drawPerformanceLine = drawPerformanceLine
        self.drawingResultantSubgraph = drawingResultantSubgraph
        self.dataPath = dataPath
        self.nubOfSubj = nubOfSubj
        self.drawEachSub = drawEachSub

    def __call__(self, dataSeries,  measure, color, picName):
        if not os.path.exists(os.path.join(self.dataPath, measure)):
            os.makedirs(os.path.join(self.dataPath, measure))
        # columnName = dataSeries.index.get_level_values(0).name
        numColumns = dataSeries.index.get_level_values(0).nunique()
        # rowName = dataSeries.index.get_level_values(1).name
        numRows = dataSeries.index.get_level_values(1).nunique()
        conditionOrderLevel0 = dataSeries.index.get_level_values(0).unique().tolist()
        drawDictLevel0 = {condition: dataSeries.groupby(level=0).get_group(condition) for condition in conditionOrderLevel0}
        self.drawingResultantSubgraph(drawDictLevel0, numRows, numColumns, measure, color, nubOfSubj, drawPerformanceLine, picName + '.png', dataPath)

        if self.drawEachSub:
            nameList = dataSeries.index.get_level_values(3).unique().tolist()
            drawNameDict = {name: dataSeries.groupby(level=3).get_group(name) for name in nameList}
            for name, seriesEachSub in drawNameDict.items():
                drawDictLevel0 = {condition: seriesEachSub.groupby(level=0).get_group(condition) for condition in conditionOrderLevel0}
                self.drawingResultantSubgraph(drawDictLevel0, numRows, numColumns, measure, color, nubOfSubj, drawPerformanceLine, picName + '_{}'.format(name) + '.png', dataPath)


if __name__ == '__main__':
    manipulatedVariables = OrderedDict()
    # manipulatedVariables['trainCondition'] = ['Noleash', 'SchematicWithReactionForce', 'Physics']
    manipulatedVariables['trainCondition'] = ['withoutReactionForce','withReactionForce']
    manipulatedVariables['lineConnection'] = ['masterWolf', 'masterDistractor', 'wolfDistractor']
    manipulatedVariablesKey = [key for key in manipulatedVariables.keys()] + ['name']

    dataPath = os.path.join(DIRNAME, '..', '..', 'resultsLocalDiffConDetectChaseWithRFAndOri1.23')
    pngFiles = glob.glob(os.path.join(dataPath, "*.png"))
    # # 遍历所有png文件并删除
    # for file in pngFiles:
    #     os.remove(file)
    rawdataSeries = pd.concat(map(pd.read_csv, glob.glob(os.path.join(dataPath, 'local_data.csv'))))
    analyzeLevelList = [f'expansionColumn{i + 1}' for i in range(4 - len(manipulatedVariablesKey))] + manipulatedVariablesKey
    rawdataSeries = rawdataSeries.assign(**{f'expansionColumn{i + 1}': 'padding' for i in range(4 - len(manipulatedVariablesKey))})
    nubOfSubj = len(rawdataSeries["name"].unique())
    drawEachSub = True
    rawdataSeries.rename(columns={'lineCondition': 'lineConnection'}, inplace=True)
    rawdataSeries['trainCondition'] = rawdataSeries.apply(lambda row: manipulatedVariables['trainCondition'][row['rf']], axis=1)

    dataDetectionDf = rawdataSeries
    dataDetectionDf['detection'] = dataDetectionDf.apply(lambda row: 1 if (row['hideId'] == 3 and row['ifJudgeChase'] == 1) or (row['hideId'] == 1 and row['ifJudgeChase'] == 0) else 0, axis=1)
    # print(dataDetectionDf)
    # 筛选有追逐的trial
    dataIdentificationDf = rawdataSeries.loc[rawdataSeries["hideId"].isin([3])]
    dataIdentificationDf["identification"] = dataIdentificationDf.apply(lambda row: 1 if (row['ifJudgeChase'] == 1 and row['selectWolf'] == 0 and row['selectSheep'] == 1) else 0, axis=1)

    dataDetectionDfMean = dataDetectionDf.groupby(analyzeLevelList)['detection'].mean()
    dataIdentificationDfMean = dataIdentificationDf.groupby(analyzeLevelList)['identification'].mean()
    # print(dataIdentificationDfMean)
    dataDetectionDfMean = dataDetectionDfMean.reindex(manipulatedVariables['trainCondition'], level='trainCondition')
    dataDetectionDfMean = dataDetectionDfMean.reindex(manipulatedVariables['lineConnection'], level='lineConnection')
    dataIdentificationDfMean = dataIdentificationDfMean.reindex(manipulatedVariables['trainCondition'], level='trainCondition')
    dataIdentificationDfMean = dataIdentificationDfMean.reindex(manipulatedVariables['lineConnection'], level='lineConnection')

    # print(dataDetectionDfMean)
    # analyzeDiffTrainConMasterWolf = AnalyzeDiffTrainConMasterWolf(independentSampleTtest)
    # result = analyzeDiffTrainConMasterWolf(dataDetectionDfMean, manipulatedVariables['trainCondition'])
    drawDiffConResults = DrawDiffConResults(drawPerformanceLine, drawingResultantSubgraph, dataPath, nubOfSubj, drawEachSub)
    drawDiffConResults(dataDetectionDfMean, 'detection', 'red', 'detection/DetectionWithDiffConnection')
    drawDiffConResults(dataIdentificationDfMean, 'identification', 'green', 'identification/IdentificationWithDiffConnection')

    # detectionSavePath = os.path.join(dataPath, 'Exp4detection.csv')
    # identificationSavePath = os.path.join(dataPath, 'Exp4identification.csv')
    # dataDetectionDfMean.to_csv(detectionSavePath, index=True)
    # dataIdentificationDfMean.to_csv(identificationSavePath, index=True)