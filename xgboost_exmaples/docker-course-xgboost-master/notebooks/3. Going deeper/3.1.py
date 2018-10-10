
import xgboost as xgb
import seaborn as sns
import pandas as pd

sns.set(font_scale = 1.5)

dtrain = xgb.DMatrix('../data/agaricus.txt.train')
dtest = xgb.DMatrix('../data/agaricus.txt.test')


# specify training parameters
params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5

# see how does it perform
watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
bst = xgb.train(params, dtrain, num_rounds, watchlist)

trees_dump = bst.get_dump(fmap='../data/featmap.txt', with_stats=True)

for tree in trees_dump:
    print(tree)

xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')

xgb.plot_importance(bst)

importances = bst.get_fscore()
print(importances)

# create df
importance_df = pd.DataFrame({
        'Splits': list(importances.values()),
        'Feature': list(importances.keys())
    })
importance_df.sort_values(by='Splits', inplace=True)
importance_df.plot(kind='barh', x='Feature', figsize=(8,6), color='orange')













