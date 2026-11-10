import pytest
import numpy as np
from src.ml_library.metrics.classification_metrics import *
from src.ml_library.metrics.classification_metrics import _calculate_auc_integral

def test_confusion_matrix():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    conf_matrix = confusion_matrix(y_true, y_pred, ['A','B','C'])

    np.testing.assert_allclose(conf_matrix, [[2, 0, 0],[1, 2, 0],[0, 1, 2]])

def test_accuracy():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    accuracy_value = accuracy(y_true, y_pred)

    assert accuracy_value == pytest.approx(0.75)

def test_precision():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    precision_array = precision(y_true, y_pred, ['A','B','C'])

    assert precision_array == pytest.approx([2/3, 2/3, 1])

def test_recall():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    recall_array = recall(y_true, y_pred, ['A','B','C'])

    assert recall_array == pytest.approx([1, 2/3, 2/3])

def test_f1_score():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    f1_score_array = f1_score(y_true, y_pred, ['A','B','C'])

    assert f1_score_array == pytest.approx([4/5, 2/3, 4/5])

def test_f1_score():
    y_pred = ['A','A','A','B','B','B','C','C']
    y_true = ['A','A','B','B','B','C','C','C']

    specificity_array = specificity(y_true, y_pred, ['A','B','C'])

    assert specificity_array == pytest.approx([5/6, 4/5, 1])

def test_log_loss():
    y_true = ['A', 'B', 'C']

    y_pred = [[0.85, 0.10, 0.05],
            [0.10, 0.70, 0.20],
            [0.05, 0.05, 0.90]]

    log_loss_value = log_loss(y_true, y_pred, ['A','B','C'])

    assert log_loss_value == pytest.approx(0.2081847)

def test_area_under_curve():
    y_true = np.array(['A', 'A', 'B', 'B', 'C'])

    y_prob = np.array([
        [0.6, 0.3, 0.1],
        [0.4, 0.5, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6]
    ])

    auc_ovr_macro = area_under_curve(y_true, y_prob, ['A','B','C'], method='OvR', weighted=False)
    auc_ovr_weighted = area_under_curve(y_true, y_prob, ['A','B','C'], method='OvR', weighted=True)
    auc_ovo_macro = area_under_curve(y_true, y_prob, ['A','B','C'], method='OvO', weighted=False)
    auc_ovo_weighted = area_under_curve(y_true, y_prob, ['A','B','C'], method='OvO', weighted=True)

    assert auc_ovr_macro == pytest.approx(0.972222)
    assert auc_ovr_weighted == pytest.approx(0.966667)
    assert auc_ovo_macro == pytest.approx(0.979167)
    assert auc_ovo_weighted == pytest.approx(0.975000)

def test_calculate_auc_integral():
    y_true = np.array(['A','A','B','B','C','B','C'])

    y_prob = np.array([
        [0.6, 0.3, 0.1],
        [0.4, 0.5, 0.1],
        [0.2, 0.7, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.3, 0.6],
        [0.8, 0.1, 0.1],
        [0.3, 0.4, 0.3]
    ])

    partition = np.linspace(0,1,20)[::-1]

    auc_a = _calculate_auc_integral(y_true,y_prob[:,0],partition,'A')
    auc_b = _calculate_auc_integral(y_true,y_prob[:,1],partition,'B')
    auc_c = _calculate_auc_integral(y_true,y_prob[:,2],partition,'C')

    assert auc_a == pytest.approx(0.8)
    assert auc_b == pytest.approx(0.625)
    assert auc_c == pytest.approx(1.0)

def test_exceptions():
    with pytest.raises(TypeError) as excep:
        accuracy(1, [1, 2, 3])
    assert str(excep.value) == "Invalid data type: Expected the target true values (y_true) to be in a list format."

    with pytest.raises(TypeError) as excep:
        accuracy([1, 2, 3], 1)
    assert str(excep.value) == "Invalid data type: Expected the predicted values (y_pred) to be in a list format."

    with pytest.raises(ValueError) as excep:
        accuracy([1, 2, 3], [1,2])
    assert str(excep.value) == "The number of empirical values (y_true) must equal the number of predicted values (y_pred)."

    with pytest.raises(TypeError) as excep:
        precision([1,2,3],[1,2,3], 1)
    assert str(excep.value) == "Invalid data type: Expected the labels parameter to be in a list format."

    with pytest.raises(ValueError) as excep:
        precision([1,2,3,4],[1,2,3,1],[1,2,3])
    assert str(excep.value) == "The classes in the true values of y (y_true) must match the classes specified in labels parameter."

    with pytest.raises(ValueError) as excep:
        precision([1,2,3,1],[1,2,3,4],[1,2,3])
    assert str(excep.value) == "The classes in the predicted values of y (y_pred) must match the classes specified in labels parameter."

    with pytest.raises(TypeError) as excep:
        p_matrix = np.array([['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']])
        log_loss([1,2,3], p_matrix, [1,2,3])
    assert str(excep.value) == f"Invalid data type: Expected probability matrix to have a numeric data type, got '{p_matrix.dtype}'"

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([0.1, 0.2, 0.7])
        log_loss([1,2,3], p_matrix, [1,2,3])
    assert str(excep.value) == "Expected a 2-D array for the probability matrix."

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[0.1, 0.9],[0.6, 0.6]])
        log_loss([1,1], p_matrix, [1,2])
    assert str(excep.value) == "The probability matrix must contain only non-negative numbers and each row must sum to 1."

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[1.2, -0.2],[0.4, 0.6]])
        log_loss([1,1], p_matrix, [1,2])
    assert str(excep.value) == "The probability matrix must contain only non-negative numbers and each row must sum to 1."

    with pytest.raises(TypeError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        log_loss([1,1], p_matrix, 1)
    assert str(excep.value) == "Invalid data type: Expected the labels parameter to be in a list format."

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        log_loss([1,3], p_matrix, [1,2])
    assert str(excep.value) == "The classes in the true values of y (y_true) must match the classes specified in labels parameter."

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        log_loss([1], p_matrix, [1,2])
    assert str(excep.value) == f"The shape of the probability matrix must be (n_points, n_labels), got '{p_matrix.shape}'"

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        log_loss([1,2], p_matrix, [1,2,3])
    assert str(excep.value) == f"The shape of the probability matrix must be (n_points, n_labels), got '{p_matrix.shape}'"

    with pytest.raises(ValueError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        area_under_curve([1,2], p_matrix, [1,2], method='RvO')
    assert str(excep.value) == "The method must be 'OvR' (One vs Rest) or 'OvO' (One vs One)."

    with pytest.raises(TypeError) as excep:
        p_matrix = np.array([[0.8, 0.2],[0.4, 0.6]])
        area_under_curve([1,2], p_matrix, [1,2], method='OvO', weighted='yes')
    assert str(excep.value) == "The weighted parameter must have a bool type."
