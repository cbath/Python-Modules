# https://github.com/ddaskan/customLibraries/blob/master/ml_pack.py

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

def cap_feature(df, col_name, min_val, max_val):
    df.loc[df[col_name] > max_val, col_name] = max_val
    df.loc[df[col_name] < min_val, col_name] = min_val
    return df

def downsample_balance_data(df, target_col, r=1):
    import pandas as pd
    df_major = df[df[target_col] == 0]
    df_minor = df[df[target_col] == 1]
    df_downsample_major = df_major.sample(n=int(len(df_minor)*r), replace=True)
    df_balance = pd.concat([df_minor, df_downsample_major])
    df_balance = df_balance.reset_index(drop=True)
    return df_balance

def splitFeatureTarget(df, feature_list, target):
    '''Create X, y array before modeling.
    Args:
        df : dataframe before feature/target split.
        feature_list : list of features including ids.
        target : target list
    Return:
        X, y array type
    '''
    X = df[feature_list].values
    y = df[target].values.ravel()
    return X, y

def raceClassifications(X, Y, cla_models, cla_scorings, pipeline=False, n_splits=5, save_fig=False, n_jobs=-1, stdout=True):
    """
    To race the given regression models in terms of given scores,
    X: features - array format
    Y: target variables - array format
    cla_models: regression models to race as a list, options;
    cla_scorings: classification scorings to measure as a list, options; ['ACC']
    pipeline: string, ['STD', 'MnMx', 'MxAbs', 'NRM']
    n_spilts: split size in fold
    save_fig: string, name extension to save plots to the working directory
    stdout: boolen,
    returns None
    """
    from matplotlib import pyplot
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC, LinearSVC
    #from xgboost import XGBClassifier
    from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer
    from sklearn.pipeline import Pipeline

    import time
    start = time.time()

    # prepare models
    models = []
    # non-linear models
    if 'SVC' in cla_models: models.append(('SVC', SVC()))
    if 'KNC' in cla_models: models.append(('KNC', KNeighborsClassifier()))
    if 'DTC' in cla_models: models.append(('DTC', DecisionTreeClassifier()))
    # ensembles
    #if 'XGB' in cla_models: models.append(('XGB', XGBClassifier()))
    if 'AB' in cla_models: models.append(('AB', AdaBoostClassifier()))
    if 'GBM' in cla_models: models.append(('GBM', GradientBoostingClassifier()))
    if 'RF' in cla_models: models.append(('RF', RandomForestClassifier()))
    if 'ET' in cla_models: models.append(('ET', ExtraTreesClassifier()))
    # linear models
    if 'LR' in cla_models: models.append(('LR', LogisticRegression()))
    if 'NB' in cla_models: models.append(('NB', GaussianNB()))
    if 'LDA' in cla_models: models.append(('LDA', LinearDiscriminantAnalysis()))
    if 'LSVC' in cla_models: models.append(('LSVC', LinearSVC()))

    pipelines = []
    if pipeline == 'STD': scalerobject, scalername = StandardScaler(), 'SdScaled'
    if pipeline == 'MnMx': scalerobject, scalername = MinMaxScaler(), 'MnMxScaled'
    if pipeline == 'MxAbs': scalerobject, scalername = MaxAbsScaler(), 'MxAbsScaled'
    if pipeline == 'NRM': scalerobject, scalername = Normalizer(), 'Normalized'

    if pipeline:
        for i in models:
            pipelines.append((scalername + i[0], Pipeline([(scalername, scalerobject), i])))
        models = pipelines

    # evaluate each model in turn
    results = []
    names = []
    scorings = []
    if 'ACC' in cla_scorings: scorings.append(('ACC', 'accuracy'))
    if 'F1' in cla_scorings: scorings.append(('F1', 'f1'))
    if 'AUC' in cla_scorings: scorings.append(('AUC', 'roc_auc'))

    out_data = {'Algo Abbr.':[], 'Mean of Scores':[], 'STD of Scores':[], 'Type of Score':[], 'Process Time':[]}
    if stdout: print("Algo Abbr.:", "Mean of Scores -", "(STD of Scores) -", "Type of Score -", "Model Process Time in Sec.")
    for scoring_name, scoring in scorings:
        for name, model in models:
            model_start = time.time()
            kfold = KFold(n_splits=n_splits, random_state=7, shuffle=True) # greater split size increase the variance in the results
            cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring, n_jobs=n_jobs)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f - (%f) - %s" % (name, cv_results.mean(), cv_results.std(), scoring_name)
            elp_time = round(time.time() - model_start, 2)
            if stdout: print(msg, "-", elp_time)
            out_data['Algo Abbr.'].append(name)
            out_data['Mean of Scores'].append(cv_results.mean())
            out_data['STD of Scores'].append(cv_results.std())
            out_data['Type of Score'].append(scoring_name)
            out_data['Process Time'].append(elp_time)

    # boxplot algorithm comparison
    first_index = 0
    for scoring_name, scoring in scorings:
        fig = pyplot.figure()
        fig.suptitle('Algorithm Comparison in terms of ' + scoring_name)
        ax = fig.add_subplot(111)
        pyplot.boxplot(results[first_index : first_index + len(models)])
        ax.set_xticklabels(names[first_index : first_index + len(models)])
        if save_fig:
            fig.savefig('algo_comp_' + save_fig + '_' + scoring_name)
        if pipeline: pyplot.xticks(rotation=35)
        if stdout:
            pyplot.show()
        else:
            pyplot.close(fig)
        first_index = first_index + len(models)

    if stdout: print("Done in " + str(time.time() - start), "seconds.")
    return pd.DataFrame(out_data)
