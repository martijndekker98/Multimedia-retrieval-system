import vedo
from distanceFunctions import findMostSimilar
import step4
from collections import defaultdict
import sys
from ann import query_kdtree
import numpy as np
import pickle

def metrics(confusion_matrix, category): # Berekent de tp fp fn tn

    tp = confusion_matrix[category, category]
    total_value = np.sum(confusion_matrix)
    # fp = np.sum(confusion_matrix[category,:]) - tp
    fp = np.sum(confusion_matrix, axis=1)[category] - tp
    # fn = np.sum(confusion_matrix[:,category]) - tp
    fn = np.sum(confusion_matrix, axis=0)[category] - tp
    tn = total_value - fp - fn - tp

    return tp, fp, fn, tn

def precision(metric): # Berekent de precision, recall en f1 op basis van de tp fp fn tn

    tp = metric[0]
    fp = metric[1]
    fn = metric[2]
    tn = metric[3]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def loop(v): # main loop van het berekenen van de metrics (zonder mean average precision)
    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')
    with open('distances.pkl', 'rb') as f:
        lookup_list = pickle.load(f)

    confusion_matrix_custom = np.zeros((19,19), dtype=np.int16)
    confusion_matrix_knn = np.zeros((19,19), dtype=np.int16)
    category_list=list(dataframe.iloc[:,0].unique())

    precision_list = []
    # print(confusion_matrix_custom)

    # modelsNegated = dataframe.loc[dataframe['File_name'] != m1_name]
    for index, row in dataframe.iterrows(): # Voor elke shape in de database doe:
        # print(index)

        # dataframe_np = dataframe.to_numpy()
        # dataframe_np_head = dataframe_np
        # dataframe_np = dataframe_np[:,2:]
        category = row.iloc[0]
        file_name= row.iloc[1]

        path_file = f"DB_fixed/{category}/{file_name}"

        results_knn = query_kdtree(dataframe, file_name, 1, v) # Query op basis van knn
        # results_custom = lookup_list[index][:10]

        confusion_column = category_list.index(category) # Label index

        for i in range(len(results_knn)): # Voor elke query result doe:

            name_knn = results_knn[i][0]
            resultaat_knn = dataframe.loc[dataframe['File_name'] == name_knn]
            folder_knn = resultaat_knn['Subfolder'].tolist()[0]
            index_knn = category_list.index(folder_knn)
            confusion_matrix_knn[index_knn, confusion_column] += 1

            # name_custom = results_custom[i][0]
            # resultaat_custom = dataframe.loc[dataframe['File_name'] == name_custom]
            # folder_custom = resultaat_custom['Subfolder'].tolist()[0]

            # index_custom = category_list.index(folder_custom) # prediction index

            # confusion_matrix_custom[index_custom, confusion_column] += 1 # verhoog confustion matrix op de goede plek


    avg_precision = [0 for i in range(19)]
    avg_recall = [0 for i in range(19)]
    avg_f1 = [0 for i in range(19)]
    for i in range(19): # Voor elke categorie doe:
        tp, fp, fn, tn = metrics(confusion_matrix_knn, i) # Bereken metrics per class
        precision_, recall_, f1_ = precision([tp, fp, fn, tn]) # Bereken precision etc per class
        avg_precision[i] += precision_ # Voeg waarden toe aan de average waarde
        avg_recall[i] += recall_
        avg_f1[i] += f1_
        precision_list.append([precision_, recall_, f1_])


    avg_precision = [i for i in avg_precision] 

    avg_recall = [i for i in avg_recall] 
    avg_f1 = [i for i in avg_f1] 

    
    # print(avg_precision, avg_recall, avg_f1)
    return avg_precision
    # return avg_precision, avg_recall, avg_f1 # Return de average precision voor de mean average precision

def mean_average_precision_custom():
    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')

    confusion_matrix_custom = np.zeros((19,19), dtype=np.int16)
    confusion_matrix_knn = np.zeros((19,19), dtype=np.int16)
    category_list=list(dataframe.iloc[:,0].unique())

    # precision_list = []
    # # print(confusion_matrix_custom)

    # # modelsNegated = dataframe.loc[dataframe['File_name'] != m1_name]

    # lookup_list = []
    # for index, row in dataframe.iterrows(): # Voor elke shape in de database doe:
    #     print(index)

    #     dataframe_np = dataframe.to_numpy()
    #     dataframe_np_head = dataframe_np
    #     dataframe_np = dataframe_np[:,2:]
    #     category = row.iloc[0]
    #     file_name= row.iloc[1]

    #     path_file = f"DB_fixed/{category}/{file_name}"

    #     results_custom_19 = findMostSimilar(dataframe, file_name, 19)
    #     lookup_list.append(results_custom_19)

    # # print(lookup_list)
    # with open('distances.pkl', 'wb') as f:
    #     pickle.dump(lookup_list, f)

    with open('distances.pkl', 'rb') as f:
        lookup_list = pickle.load(f)

    # mean_a_p_list = 0
    mean_a_p_list = [0 for i in range(19)]
    for iteration in range(1,20):
        print(iteration)
        for index, row in dataframe.iterrows(): # Voor elke shape in de database doe:
            # print(index)
            category = row.iloc[0]
            results = lookup_list[index][:iteration]
            confusion_column = category_list.index(category) # Label index

            for i in range(len(results)): # Voor elke query result doe:

                name_custom = results[i][0]
                resultaat_custom = dataframe.loc[dataframe['File_name'] == name_custom]
                folder_custom = resultaat_custom['Subfolder'].tolist()[0]

                index_custom = category_list.index(folder_custom) # prediction index

                confusion_matrix_custom[index_custom, confusion_column] += 1 # verhoog confustion matrix op de goede plek

        # avg_precision = 0
        # avg_precision_list  = [0 for i range(19)]

        for i in range(19): # Voor elke categorie doe:
            tp, fp, fn, tn = metrics(confusion_matrix_custom, i) # Bereken metrics per class
            precision_, recall_, f1_ = precision([tp, fp, fn, tn]) # Bereken precision etc per class
            mean_a_p_list[i] += precision_
 
        # avg_precision = avg_precision / 19 # Deel door aantal categoriën 
        # avg_recall = avg_recall / 19
        # avg_f1 = avg_f1 / 19
        # print(avg_precision)
        # mean_a_p += avg_precision
    mean_a_p_list_custom = [i/19 for i in mean_a_p_list]
    print(mean_a_p_list_custom)

    return mean_a_p_list_custom


def mean_average_precision_knn():
    total_precision = [0 for i in range(19)] 
    for i in range(1, 20):
        print(i)
        avg_precision = loop(i)
        for j in range(len(total_precision)):
            total_precision[j] += avg_precision[j]
            
        # total_precision += avg_precision
    mean_avg_pr = [i/19 for i in total_precision]
    print(mean_avg_pr)
    return mean_avg_pr

def main():
    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')
    category_list=list(dataframe.iloc[:,0].unique())
    metric_value = mean_average_precision_knn()
    sum_map = np.sum(metric_value) / 19
    best_map = category_list[np.argmax(metric_value)]
    worst_map = category_list[np.argmin(metric_value)]
    print(sum_map)
    print(best_map, metric_value[np.argmax(metric_value)])
    print(worst_map, metric_value[np.argmin(metric_value)])
    # sum_precision = np.sum(metric_value[0]) / 19
    # sum_recall = np.sum(metric_value[1]) / 19
    # sum_f1 = np.sum(metric_value[2]) / 19
    # print(sum_precision, sum_recall, sum_f1)
    # print(metric_value)
    # best_precision = category_list[np.argmin(metric_value[0])]
    # best_recall = category_list[np.argmin(metric_value[1])]
    # best_f1 = category_list[np.argmin(metric_value[2])]
    # print(best_precision, metric_value[0][np.argmin(metric_value[0])])
    # print(best_recall, metric_value[1][np.argmin(metric_value[1])])
    # print(best_f1, metric_value[2][np.argmin(metric_value[2])])

    # print(loop(10)[0]) 
    # print(loop(10)[1])
    # print(loop(10)[2]) # calculates precision, recall and f1 for a single query size
    # mean_average_precision_custom() # Custom mean average precision
    # mean_average_precision_knn() # Knn mean average precision

main()
        
        

def acc(): # main loop van het berekenen van de metrics (zonder mean average precision)
    dataframe = step4.readCSVAsDataFrame('./featuresNew.csv')

    confusion_matrix_custom = np.zeros((19,19), dtype=np.int16)
    confusion_matrix_knn = np.zeros((19,19), dtype=np.int16)
    category_list=list(dataframe.iloc[:,0].unique())

    with open('distances.pkl', 'rb') as f:
        lookup_list = pickle.load(f)

    acc = 0
    for index, row in dataframe.iterrows(): # Voor elke shape in de database doe:
        # print(index)

        # dataframe_np = dataframe.to_numpy()
        # dataframe_np_head = dataframe_np
        # dataframe_np = dataframe_np[:,2:]
        category = row.iloc[0]
        file_name= row.iloc[1]

        path_file = f"DB_fixed/{category}/{file_name}"

        results_knn = query_kdtree(dataframe, file_name, 1, 1)[0] # Query op basis van knn
        results_custom = lookup_list[index][0]

        name_custom = results_knn[0]

        resultaat_custom = dataframe.loc[dataframe['File_name'] == name_custom]
        folder_custom = resultaat_custom['Subfolder'].tolist()[0]
        # print(folder_custom)


        if folder_custom == category:
            acc += 1

    print(acc/380)

# acc()



