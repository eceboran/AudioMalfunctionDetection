import random

def add_train_test_labels_to_dataframe(df, no_seed):
    # Separate data into training and test sets
    # All abnormal samples are in the test set
    # Equal number of normal samples in the test set
    # Remaining in the training set
    no_abnormal_test = sum(df.anomaly==1)
    no_normal_test = no_abnormal_test
    no_normal_training = df.shape[0]-no_normal_test-no_abnormal_test

    df['test_train'] = 0
    df.loc[df.anomaly==1, 'test_train'] = 1
    random.seed(no_seed)
    ind_normal_test = random.sample(list(df[df.anomaly==0].index), no_normal_test)
    df.loc[ind_normal_test, 'test_train'] = 1

    df.groupby(['anomaly','test_train']).count()
    
    return df