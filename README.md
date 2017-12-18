## Solution

Rank 1: https://www.kaggle.com/c/kkbox-music-recommendation-challenge/discussion/45942

## MISC

FE, LGB, 60 features

CF: no use, too sparse



 network/graph embedding and matrix factorization



1) raw features

2) statistics of features 

3) user-item interaction features 

4) score features computed via the learned representations



CF, at least the simple/common approaches, don't make any use of the metadata about songs and users - just user/song pairings. So LGBM has a lot of additional data to generalize from.

## Embedding

**score: 0.62582**

## Average

**score: 0.69075, rank: 114**

**score: 0.68765, rank: 161**

**score: 0.68381, rank: 266**

```
import lightgbm as lgb

d_train_final = lgb.Dataset(X_train, y_train)
watchlist_final = lgb.Dataset(X_train, y_train)

params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'dart',
        'learning_rate': 0.2,
        'verbose': 0,
        'num_leaves': 2**8,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 256,
        'max_depth': 20,
        'num_rounds': 200,
        'metric' : 'auc'
    }

model_f2 = lgb.train(params, train_set=d_train_final,  valid_sets=watchlist_final, verbose_eval=5)
```

```python
print('Making predictions')
p_test_1 = model_f1.predict(X_test)
p_test_2 = model_f2.predict(X_test)
p_test_avg = np.mean([p_test1, p_test2], axis = 0)
```

## KFold

```python
from sklearn.model_selection import KFold
import lightgbm as lgb

kf = KFold(n_splits=3)

predictions = np.zeros(shape=[len(df_test)])

for train_indices,val_indices in kf.split(df_train) : 
    train_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[train_indices,:],label=df_train.loc[train_indices,'target'])
    val_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[val_indices,:],label=df_train.loc[val_indices,'target'])
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting': 'gbdt',
        'learning_rate': 0.1 ,
        'verbose': 0,
        'num_leaves': 108,
        'bagging_fraction': 0.95,
        'bagging_freq': 1,
        'bagging_seed': 1,
        'feature_fraction': 0.9,
        'feature_fraction_seed': 1,
        'max_bin': 128,
        'max_depth': 10,
        'num_rounds': 200,
        'metric' : 'auc',
        } 
    
    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])
    predictions+=bst.predict(df_test.drop(['id'],axis=1))
    del bst
    
predictions = predictions/3
```

## MISC

If we had only msno and song_id data, matrix factorization would be much better than LightGBM. But we have lot of information about users and songs, which matrix factorization cannot contain (as far as I know). I tried MF and was capable of getting only 0.6 AUC on validation set.

So it is not worse or better overall. LGBM is better for these information about songs and users and worse on song-user id pairs.

```
pip install pandas==0.19
```

