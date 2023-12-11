from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def metrics(true_labels , thief_preds: list[int], victim_preds: list[int]):
    """
    Returns accuracy, f1, precision, recall for the model on the data_loader
    """
    assert len(thief_preds) == len(victim_preds), "Label lists must have the same length"    
    correct = sum(thief_label == victim_label for thief_label, victim_label in zip(thief_preds, victim_preds))
    total_samples = len(thief_preds)
    agreement_score = correct / total_samples
    metric = {
        'accuracy': accuracy_score(true_labels, thief_preds),
        'f1': f1_score(true_labels, thief_preds, average='macro'),
        'precision': precision_score(true_labels, thief_preds, average='macro'),
        'recall': recall_score(true_labels, thief_preds, average='macro'),
        'agreement_score': agreement_score
    }
    return metric