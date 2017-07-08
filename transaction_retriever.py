import numpy as numpy
import pandas as pd

df = pd.read_csv('transaction.csv')
labels = set(df['company label'])
salary_segmentation = [10000,50000,100000,500000,1000000]




# return personal info for each category
# [{'amount': 922.05999999999995, 'drilldown': 'Grocery', 'name': 'Grocery'}, {...}]
def getPersonalAnalysis(name):

    personal = df.loc[df['name'] == name, ['company name', 'company label', 'amount']]
    output = []
    for label in labels:
        amount = sum(personal.loc[personal['company label'] == label,'amount'])
        output.append({'name': label, 'y': amount, 'drilldown':label})
    return output
# getPersonalAnalysis('Rachel Trujillo')


# Intput:
# criteria: a list ['gender', 'age','salary']
# target: if == 'category', only return average spending for each category
#
# return: [{'amount': 154.95960794655397, 'drilldown': 'Grocery', 'name': 'Grocery'}, {...}]
#		  if == 'percentage', return total number, percentage and index in the group, and the list of amount of spending for
#                each people in each category
# return:{'total_people': 500, education': {'index': 400, 'amount_list': [...], 'percentage': 0.8}}
def getComparison(name, criteria , target='category'):
    personal = df.loc[df['name'] == name, ['age','salary','gender','company label', 'amount']]
    age = personal.iloc[0]['age']
    salary = personal.iloc[0]['salary']
    gender = personal.iloc[0]['gender']
    selectedDf = df
    criteria = set(criteria)
    if 'gender' in criteria:
        selectedDf = selectedDf.loc[selectedDf['gender'] == gender]
    if 'salary' in criteria:
        lowThresh = None
        highThresh = None

        for s in salary_segmentation:
#             print (s)
            if salary < s:
#                 print (s)
                if s == salary_segmentation[0]:
                    highThresh = s
                elif s == salary_segmentation[-1]:
                    lowThresh = s
                else:
                    index = salary_segmentation.index(s)
                    lowThresh = salary_segmentation[index-1]
                    highThresh = salary_segmentation[index]
                break
        print (lowThresh)
        print(highThresh)
        if lowThresh is not None:
            print ('fff')
            selectedDf = selectedDf.loc[selectedDf['salary'] >= lowThresh]
        if highThresh is not None:
            selectedDf = selectedDf.loc[selectedDf['salary'] < highThresh]
    if 'age' in criteria:
        lowThresh = age-age%10
        highThresh = lowThresh+9
        selectedDf = selectedDf.loc[selectedDf['age'] >= lowThresh]
        selectedDf = selectedDf.loc[selectedDf['age'] < highThresh]
    output = []
    if target =='percentage':
    	output = {}
    num_people= len(set(selectedDf.name))
    for label in labels:

        if target == 'category':
            amount = sum(selectedDf.loc[selectedDf['company label'] == label,'amount'])/len(selectedDf.loc[selectedDf['company label'] == label,'amount'])
            output.append({'name': label, 'amount': amount, 'drilldown':label})

        elif target == 'percentage':

            output['total_people'] = num_people
            amountList = []
            for name in set(selectedDf.name):

                thatPersonDf = selectedDf.loc[selectedDf['name'] == name]
    #             print(thatPersonDf)
    #             print (thatPersonDf.loc[thatPersonDf['company label'] == label, 'amount'].dtype)
                amount = (thatPersonDf.loc[thatPersonDf['company label'] == label, 'amount']).sum()
                amountList.append(amount)
            amountList.sort()

            thisPersonAmount = sum(personal.loc[personal['company label'] == label, 'amount'])
            index = amountList.index(thisPersonAmount)
            output[label] = {'index':index, 'percentage': (index+0.0)/ num_people, 'amount_list':amountList}
    print ('loaded')
    return output

# print getComparison('Rachel Trujillo', ['gender','age'], target='percentage')
