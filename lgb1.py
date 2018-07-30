 # Import need libraries
import numpy as np 
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

# Given data set, lots of modeling accuracy is derived from pre-processing data. This routine builds
# off of a few different pre-processing kernals that seeks to one-hot encode
# categorical attributes and develop interaction effects when necessary.
def preprocess(df):
    """
    Main feature engineering function.
    """
    def mk_categoricals(df, prefixes=None, subsets=None):
        """
        Converts one-hot-encoded categorical to true categorical.
        prefixes: list of prefixes of one-hot-encoded categorical variables
                  e.g. for variables
                      abastaguadentro, =1 if water provision inside the dwelling
                      abastaguafuera, =1 if water provision outside the dwelling
                      abastaguano, =1 if no water provision
                  we provide prefix "abastagua"
        subsets: dictionary {name_of_feature: [columns], ...}
                 e.g. for variables
                     public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC"
                     planpri, =1 electricity from private plant
                     noelec, =1 no electricity in the dwelling
                     coopele, =1 electricity from cooperative
                 we provide {"electricity": ['public', 'planpri', 'noelec', 'coopele']}
        """
        def mk_category(dummies):
            assert (dummies.sum(axis=1) <= 1).all()
            nans = dummies.sum(axis=1) != 1
            if nans.any():
                dummies = dummies.assign(_na=nans.astype(int))
            return dummies.idxmax(axis=1).astype('category')

        categoricals = pd.DataFrame()

        if prefixes:
            for prefix in prefixes:
                columns = df.columns[df.columns.str.startswith(prefix)]
                categoricals[prefix] = mk_category(df[columns])
        if subsets:
            for feature_name, subset in subsets.items():
                categoricals[feature_name] = mk_category(df[subset])

        return categoricals
    groupper = df.groupby('idhogar')
    interactions = (pd.DataFrame(dict(
                    head_escolari=df.parentesco1 * df.escolari,
                    head_female=df.parentesco1 * df.female,
                    head_partner_escolari=df.parentesco2 * df.escolari))
                    .groupby(df.idhogar)
                    .max())
    # basic interaction features
    
    my_features = (groupper.mean()[['escolari', 'age', 'hogar_nin', 
                                    'hogar_total', 'epared3', 'epared1',
                                    'etecho3', 'etecho1', 'eviv3', 'eviv1',
                                    'male',
                                    'r4h1', 'r4h2', 'r4h3', 'r4m1', 'r4m2', 
                                    'r4m3',
                                    'r4t1', 'r4t2', 'r4t3', 'v2a1', 'rooms', 
                                    'bedrooms',
                                    'meaneduc', 
                                    'SQBdependency', 'rez_esc', 'refrig', 
                                    'tamviv', 'overcrowding']]
                   .join(groupper.std()[['escolari', 'age']], 
                         rsuffix='_std')
                   .join(groupper[['escolari', 'age']].min(), rsuffix="_min")
                   .join(groupper[['escolari', 'age']].max(), rsuffix="_max")
                   .join(groupper[['dis']].sum(), rsuffix="_sum")
                   # partially based on
                   # https://www.kaggle.com/taindow/predicting-poverty-levels-with-r
                   .assign(child_rate=lambda x: x.hogar_nin / x.hogar_total,
                           wrf=lambda x: x.epared3 - x.epared1 +
                                         x.etecho3 - x.etecho1 +
                                         x.eviv3 - x.eviv1,
                           # wrf is an integral feature that measure
                           # quality of the house
                           escolari_range=lambda x: x.escolari_max - x.escolari_min,
                           age_range=lambda x: x.age_max - x.age_min,
                           rent_per_individual=lambda x: x.v2a1 / x.r4t3,
                           rent_per_child=lambda x: x.v2a1 / x.r4t1,
                           rent_per_over65=lambda x: x.v2a1 / x.r4t3,
                           rent_per_room=lambda x: x.v2a1 / x.rooms,
                           rent_per_bedroom=lambda x: x.v2a1 / x.bedrooms,
                           rooms_per_individual=lambda x: x.rooms / x.r4t3,
                           rooms_per_child=lambda x: x.rooms / x.r4t1,
                           bedrooms_per_individual=lambda x: x.bedrooms / x.r4t3,
                           bedrooms_per_child=lambda x: x.bedrooms / x.r4t1,
                           years_schooling_per_individual=lambda x: x.escolari / x.r4t3,
                           years_schooling_per_adult=lambda x: x.escolari / (x.r4t3 - x.r4t1),
                           years_schooling_per_child=lambda x: x.escolari / x.r4t3
                          )
                   .drop(['hogar_nin', 'hogar_total', 'epared3', 'epared1',
                                   'etecho3', 'etecho1', 'eviv3', 'eviv1'], 
                         axis=1)
                   .join(interactions)
                   .join(groupper[['computer', 'television', 
                                   'qmobilephone', 'v18q1']]
                         .mean().sum(axis=1).rename('technics'))
                   # we provide integral technical level as a new feature 
                   .assign(technics_per_individual=lambda x: x.technics / x.r4t3,
                           technics_per_child=lambda x: x.technics / x.r4t1)
                   .join(mk_categoricals(groupper.mean(), 
                                prefixes=['lugar', 'area', 'tipovivi', 
                                          'energcocinar', 
                                          'sanitario', 'pared', 'piso',
                                          'abastagua'],
                                subsets={'electricity': ['public', 
                                                         'planpri', 
                                                         'noelec', 
                                                         'coopele']}))
                  )
    return my_features

# Read in data and set Target as dependent variable (by household)
df = pd.read_csv("../input/train.csv")
X = preprocess(df)
y = df.groupby('idhogar').Target.mean().astype(int)

# Run LGB multiple classifier with 5 folds.
clf = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)


df_test = pd.read_csv("../input/test.csv").set_index('Id')
X_test = preprocess(df_test)


kf = StratifiedKFold(n_splits=5, shuffle=True)
predicts = []
for train_index, test_index in kf.split(X, y):
    print("###")
    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            early_stopping_rounds=20)
    predicts.append(clf.predict(X_test))


# Print output in required format
predict_by_hh = pd.DataFrame(np.array(predicts).mean(axis=0).round().astype(int),
                             columns=['Target'],
                             index=X_test.index)

predict = df_test.join(predict_by_hh, on='idhogar')[['Target']].astype(int)

predict.to_csv("output.csv")

