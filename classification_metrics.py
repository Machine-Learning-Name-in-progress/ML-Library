import numpy as np

def confusion_matrix(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  conf_matrix = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true, y_pred):
    conf_matrix[label_to_idx[true]][label_to_idx[pred]] += 1

  return np.array(conf_matrix)

def accuracy(y_true: list, y_pred: list) -> float:
  return sum(true==pred for true,pred in zip(y_true,y_pred))/len(y_true)

def precision(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  true_pos = [0 for _ in range(len(labels))]
  pos = [0 for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_pos[label_to_idx[pred]] += (true==pred)
    pos[label_to_idx[pred]] += 1

  precision_list = [1 if (true_pos[i]==pos[i]) else true_pos[i]/pos[i] for i in range(len(labels))]
  return np.array(precision_list)

def recall(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  true_pos = [0 for _ in range(len(labels))]
  pos = [0 for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_pos[label_to_idx[pred]] += (true==pred)
    pos[label_to_idx[true]] += 1

  recall_list = [1 if (true_pos[i]==pos[i]) else true_pos[i]/pos[i] for i in range(len(labels))]
  return np.array(recall_list)

def f1_score(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  prec = precision(y_true, y_pred, labels)
  rec = recall(y_true, y_pred, labels)
  f1 = 2*(prec*rec)/(prec+rec)
  return f1

def specificity(y_true: list, y_pred: list, labels: list) -> np.ndarray:
  true_neg = [len(y_true) for _ in range(len(labels))]
  neg = [len(y_true) for _ in range(len(labels))]
  label_to_idx = {label : idx for idx, label in enumerate(labels)}

  for true, pred in zip(y_true,y_pred):
    true_neg[label_to_idx[pred]] -= 1
    true_neg[label_to_idx[true]] -= (true!=pred)
    neg[label_to_idx[true]] -= 1

  specificity_list = [1 if (true_neg[i]==neg[i]) else true_neg[i]/neg[i] for i in range(len(labels))]
  return np.array(specificity_list)

def log_loss(y_true: list, probability_matrix: list, labels: list) -> np.ndarray:
  label_to_idx = {label : idx for idx, label in enumerate(labels)}
  log_loss_list = [0 for _ in range(len(y_true))]
  count = 0

  for y, p_list in zip(y_true, probability_matrix):
    log_loss_list[count] = -np.log(p_list[label_to_idx[y]])
    count += 1

  return np.mean(log_loss_list)