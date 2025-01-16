from sklearn.ensemble import RandomForestClassifier

def random_forest(x, y):
    rf = RandomForestClassifier(n_estimators=100, oob_score=True,
                                random_state=123456)
    rf.fit(x, y)