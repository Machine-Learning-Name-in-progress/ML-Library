import numpy as np

def confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  """Computes the confusion matrix of a multi-class classification model.

  Args:
      y_true (list): The true classes of the target y.
      y_pred (list): The predicted classes for target y.
      labels (list): The classes of the model.

  Returns:
      np.ndarray: The confusion matrix for the model.
  """
  _exceptions(y_true=y_true, y_pred=y_pred, labels=labels)

  conf_matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true, y_pred):
    conf_matrix[label_to_idx[true]][label_to_idx[pred]] += 1

  return np.array(conf_matrix)

def accuracy(y_true: list, y_pred: list) -> float:
  """Computes the accuracy of a multi-class classification model.

  Args:
      y_true (list): The true classes of the target y.
      y_pred (list): The predicted classes for target y.

  Returns:
      float: The accuracy of the model.
    """
  _exceptions(y_true=y_true, y_pred=y_pred)
  return sum(true==pred for true,pred in zip(y_true,y_pred))/len(y_true)

def precision(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  """Computes the precision of a multi-class classification model.

  Args:
    y_true (list): The true classes of the target y.
    y_pred (list): The predicted classes for target y.
    labels (list): The classes of the model.

  Returns:
    np.ndarray: The array with the precision metric for each class of the model.
  """
  _exceptions(y_true=y_true, y_pred=y_pred, labels=labels)

  true_pos = [0 for _ in range(len(labels))]
  pos = [0 for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_pos[label_to_idx[pred]] += (true==pred)
    pos[label_to_idx[pred]] += 1

  precision_list = [1 if (true_pos[i]==pos[i]) else true_pos[i]/pos[i] for i in range(len(labels))]
  return np.array(precision_list)

def recall(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  """Computes the recall (sensitivity/true positive rate) of a multi-class classification model.

  Args:
    y_true (list): The true classes of the target y.
    y_pred (list): The predicted classes for target y.
    labels (list): The classes of the model.

  Returns:
    np.ndarray: The array with the recall metric for each class of the model.
  """
  _exceptions(y_true=y_true, y_pred=y_pred, labels=labels)

  true_pos = [0 for _ in range(len(labels))]
  pos = [0 for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_pos[label_to_idx[pred]] += (true==pred)
    pos[label_to_idx[true]] += 1

  recall_list = [1 if (true_pos[i]==pos[i]) else true_pos[i]/pos[i] for i in range(len(labels))]
  return np.array(recall_list)

def f1_score(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  """Computes the f1-score of a multi-class classification model.

  Args:
      y_true (list): The true classes of the target y.
      y_pred (list): The predicted classes for target y.
      labels (list): The classes of the model.

  Returns:
      np.ndarray: The array with the f1-score for each class of the model.
  """
  _exceptions(y_true=y_true, y_pred=y_pred, labels=labels)

  prec = precision(y_true, y_pred, labels)
  rec = recall(y_true, y_pred, labels)
  f1 = 2*(prec*rec)/(prec+rec)
  return f1

def specificity(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  """Computes the specificity (true negative rate) of a multi-class classification model.

  Args:
      y_true (list): The true classes of the target y.
      y_pred (list): The predicted classes for target y.
      labels (list): The classes of the model.

  Returns:
      np.ndarray: The array with the specificity metric for each class of the model.
  """
  _exceptions(y_true=y_true, y_pred=y_pred, labels=labels)

  true_neg = [len(y_true) for _ in range(len(labels))]
  neg = [len(y_true) for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_neg[label_to_idx[pred]] -= 1
    true_neg[label_to_idx[true]] -= (true!=pred)
    neg[label_to_idx[true]] -= 1

  specificity_list = [1 if (true_neg[i]==neg[i]) else true_neg[i]/neg[i] for i in range(len(labels))]
  return np.array(specificity_list)

def log_loss(y_true: list, probability_matrix: list, labels: list) -> float:
  """Computes the log-loss of a multi-class classification model.

  Args:
      y_true (list): The true classes of the target y.
      probability_matrix (list): The predicted probability matrix for target y.
      labels (list): The classes of the model.

  Returns:
      float: The log-loss metric of the model.
  """
  _exceptions(y_true=y_true, probability_matrix=probability_matrix, labels=labels)

  label_to_idx = {label : idx for idx, label in enumerate(labels)}
  log_loss_list = [0 for _ in range(len(y_true))]
  count = 0

  for y, p_list in zip(y_true, probability_matrix):
    log_loss_list[count] = -np.log(p_list[label_to_idx[y]])
    count += 1

  return np.mean(log_loss_list)

def area_under_curve(y_true: list, probability_matrix: list, labels: list, method: str = "OvR", weighted: bool = False) -> float:
  """Computes the area under curve metric of a multi-class classification model.
  
  Args:
      y_true (list): The true classes of the target y.
      probability_matrix (list): The predicted probability matrix for target y.
      labels (list): The classes of the model.
      method (str, optional): Define the method used to lead with multiple-class models. It can be the 'OvR' (One-vs-Rest) or 'OvO' (One-vs-One) method.
      weighted (bool, optional): Defines whether the average will be weighted by frequency of occurrency or not.

  Returns:
      float: The area under curve metric of the model.
  """
  _exceptions(y_true=y_true, probability_matrix=probability_matrix, labels=labels, method=method, weighted=weighted)

  if(method=="OvR"):
    y_true = np.asarray(y_true)
    p_matrix = np.asarray(probability_matrix)
    auc_array = []
    if weighted: weights = []

    for i, label in enumerate(labels):
      p_vector = p_matrix[:,i]
      thresholds = np.sort(p_vector)[::-1]
      auc = _calculate_auc_integral(y_true, p_vector, thresholds, label)
      auc_array.append(auc)
      if weighted: weights.append(np.sum(y_true==label))

  if(method=="OvO"):
    y_true = np.asarray(y_true)
    p_matrix = np.asarray(probability_matrix)
    auc_array = []
    if weighted: weights = []

    for i in range(len(labels)):
      for j in range(i+1, len(labels)):
        mask_ij = np.isin(y_true, [labels[i], labels[j]])
        y_true_ij = y_true[mask_ij]

        p_i = p_matrix[:,i][mask_ij]
        p_j = p_matrix[:,j][mask_ij]

        thresholds_i = np.sort(p_i)[::-1]
        thresholds_j = np.sort(p_j)[::-1]

        auc_i = _calculate_auc_integral(y_true_ij, p_i, thresholds_i, labels[i])
        auc_j = _calculate_auc_integral(y_true_ij, p_j, thresholds_j, labels[j])

        auc_array.append((auc_i+auc_j)/2)
        if weighted: weights.append(np.sum(mask_ij))

  return np.average(auc_array, weights=weights) if weighted else np.mean(auc_array)

def _calculate_auc_integral(y_true: np.ndarray, probability_array: np.ndarray, partition: np.ndarray, label: str) -> float:
  """Private method to compute the integral of the ROC curve (receiver operating characteristic curve) using the trapezoidal method.
  
  Args:
      y_true (np.ndarray): The true classes of the target y.
      probability_array (np.ndarray): The predicted probability array of the reference class for target y.
      partition (np.ndarray): The thresholds used to calculate the curve points.
      labels (str): The class of reference for the curve.

  Returns:
      float: The integral of the ROC curve.
  """
  tpr_array = []
  fpr_array = []
  for prob in partition:
    y_true_bin = (y_true==label).astype(int)
    y_pred_bin = (probability_array>=prob).astype(int)

    tpr_array.append(recall(y_true_bin, y_pred_bin, [1, 0])[0])
    fpr_array.append(1 - specificity(y_true_bin, y_pred_bin, [1, 0])[0])

  tpr_array = np.concatenate(([0], tpr_array, [1]))
  fpr_array = np.concatenate(([0], fpr_array, [1]))

  integral = np.sum((fpr_array[1:] - fpr_array[:-1])*(tpr_array[:-1] + tpr_array[1:]))/2
  return integral

def _exceptions(y_true: list, y_pred: list = None, probability_matrix: list = None, labels: list = None, method: str = None, weighted: bool = None):
  """Private method to log internal exceptions"""
  if not isinstance(y_true, (list, np.ndarray, tuple)):
     raise TypeError("Invalid data type: Expected the target true values (y_true) to be in a list format.")

  if y_pred != None:
    if not isinstance(y_pred, (list, np.ndarray, tuple)):
        raise TypeError("Invalid data type: Expected the target predicted values (y_pred) to be in a list format.")

    if len(y_true)!=len(y_pred):
        raise ValueError("The number of empirical values (y_true) must equal the number of predicted values (y_pred).")

    if labels != None:
      if not isinstance(labels, (list, np.ndarray, tuple)):
          raise TypeError("Invalid data type: Expected the labels parameter to be in a list format.")
    
      if not all(y in labels for y in y_true):
        raise ValueError("The classes in the true values of y (y_true) must match the classes specified in labels parameter.")
      
      if not all(y in labels for y in y_pred):
          raise ValueError("The classes in the predicted values of y (y_pred) must match the classes specified in labels parameter.")

    else:
      p_matrix = np.ndarray(probability_matrix)

      if not np.issubdtype(p_matrix.dtype, np.number):
         raise TypeError(f"Invalid data type: Expected probability matrix to have a numeric data type, got '{p_matrix.dtype}'")

      if(X.ndim!=2):
         raise ValueError("Expected a 2-D array for the probability matrix.")

      if not bool(np.all(p_matrix>=0) and np.allclose(np.sum(p_matrix, axis=1), 1)):
         raise ValueError("The probability matrix must contain only non-negative numbers and each row must sum to 1.")

      if not isinstance(labels, (list, np.ndarray, tuple)):
         raise TypeError("Invalid data type: Expected the labels parameter to be in a list format.")
      
      if not all(y in labels for y in y_true):
              raise ValueError("The classes in the true values of y (y_true) must match the classes specified in labels parameter.")

      if (p_matrix.shape[0]!=len(y_true)) or (p_matrix.shape[1]!=len(labels)):
         raise ValueError(f"The shape of the probability matrix must be (n_points, n_labels), got '{p_matrix.shape}'")

      if (method!=None) and (method!="OvR") and (method!="OvO"):
        raise ValueError("The method must be 'OvR' (One vs Rest) or 'OvO' (One vs One).")

      if (weighted!=None) and (not isinstance(weighted, bool)):
         raise TypeError("The weighted parameter must have a bool type.")