# Creating a function to report confusion metrics
def confusion_metrics (conf_matrix):
    
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    
    if(FP==0):
        FP = 0.000001
    if(FN==0):
        FN = 0.000001
        
    # Accuracy
    accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # Precision and recall
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    
    # Sensitivity and specificity
    sensitivity = recall
    specificity = (TN / float(TN + FP))
    
    # TPR and FPR
    TPR = recall
    FPR = 1-specificity
    
    # F_1 score
    f1 = 2 * ((precision * sensitivity) / (precision + sensitivity))
    
    out_metrics = {'acc': accuracy,
                  'precision': precision,
                  'recall': recall,
                  'sensitivity': sensitivity,
                  'specificity': specificity,
                  'TPR': TPR,
                  'FPR': FPR,
                  'f1': f1}
    
    return out_metrics