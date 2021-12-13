#%%
from hoag import LogisticRegressionCV

# load some data
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

#%%
# get a training set and test set
data_train = datasets.fetch_20newsgroups_vectorized(subset='train')
data_test = datasets.fetch_20newsgroups_vectorized(subset='test')

X_train = data_train.data
X_test = data_test.data;
y_train = data_train.target
y_test = data_test.target

# binarize labels
y_train[data_train.target < 10] = -1
y_train[data_train.target >= 10] = 1
y_test[data_test.target < 10] = -1
y_test[data_test.target >= 10] = 1

#%%  
clf = LogisticRegressionCV() #exponetial
clf.fit(X_train, y_train, X_test, y_test)
print('Regularization chosen by HOAG: alpha=%s' % (clf.alpha_[0]))

#%%
clf_cub = LogisticRegressionCV(tolerance_decrease='cubic')
clf_cub.fit(X_train, y_train, X_test, y_test)
clf_cub.alpha_[0]

#%%
clf_qua = LogisticRegressionCV(tolerance_decrease='quadratic')
clf_qua.fit(X_train, y_train, X_test, y_test)
clf_qua.alpha_[0]

#%%
clf_exa = LogisticRegressionCV(tolerance_decrease='exact')
clf_exa.fit(X_train, y_train, X_test, y_test)
clf_exa.alpha_[0]