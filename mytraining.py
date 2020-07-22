import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


if __name__ == "__main__":
    # read the data
    df=pd.read_csv("data.csv")
    train,test=train_test_split(df, test_size=0.2 , random_state=42)

    x_train=train[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()
    x_test=test[['fever', 'bodyPain', 'age', 'runnyNose', 'diffBreath']].to_numpy()

    y_train=train[['infectionProb']].to_numpy().reshape(2044,)
    y_test=test[['infectionProb']].to_numpy().reshape(511,)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)

    # close the file
    file.close()
    

