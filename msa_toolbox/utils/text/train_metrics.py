from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def simple_accuracy(preds, labels):
    '''
    Returns accuracy for the model on the data_loader
    '''
    return (preds == labels).mean()

def agreement_score(thief_labels: list[int], victim_labels: list[int]) -> float:
    '''
    Calculates the agreement between the thief and victim model based on output label lists
    '''
    assert len(thief_labels) == len(victim_labels), "Label lists must have the same length"    
    correct = sum(thief_label == victim_label for thief_label, victim_label in zip(thief_labels, victim_labels))
    total_samples = len(thief_labels)
    return correct / total_samples

def accuracy_f1_precision_recall(preds, labels):
    """
    Returns accuracy, f1, precision, recall for the model on the data_loader
    """
    metric = {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='macro'),
        'precision': precision_score(labels, preds, average='macro'),
        'recall': recall_score(labels, preds, average='macro')
    }
    return metric