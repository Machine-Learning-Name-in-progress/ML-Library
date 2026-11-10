from ml_library.models.decision_tree import decision_tree_classifier, decision_tree_regressor
import numpy as np

def test_classifier():
    tree = decision_tree_classifier(100, 2, 1)
    x = np.array([
        [1,10,1],
        [1,2,0],
        [0,0,0]
    ])
    
    y = np.array(['dog', 'cat', 'cat'])

    tree.fit(x, y)
    assert tree.predict(np.array([[1,10,1]])) == 'dog'
    assert tree.predict(np.array([[0,0,0]])) == 'cat'

def test_regressor():
    tree = decision_tree_regressor(100, 2, 1)
    x = np.array([
        [1,10,1],
        [1,2,0],
        [0,0,0]
    ])
    
    y = np.array([10, 1.4, 0.0])

    tree.fit(x, y)
    assert tree.predict(np.array([[1,10,1]])) == 10
    assert (tree.predict(np.array([[0,0,0], [1,2,0]])) == [0.0, 1.4]).all()
