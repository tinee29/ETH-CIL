import re
from collections import defaultdict
import matplotlib.pyplot as plt

#goes through log lin by line and appends (model,F1 score) touples to list
def parse_f1_scores(log_file_path):
    with open(log_file_path, 'r') as file:
        log_content = file.readlines()
    
    f1_train_scores = []
    f1_val_scores = []
    current_model = None
    
    model_pattern = re.compile(r"Training model: (\S+)")
    f1_train_pattern = re.compile(r"\s*- f1 = (\d+\.\d+)")
    f1_val_pattern = re.compile(r"\s*- val_f1 = (\d+\.\d+)")

    for line in log_content:
        model_match = model_pattern.search(line)
        f1_train_match = f1_train_pattern.search(line)
        f1_val_match = f1_val_pattern.search(line)
        
        if model_match:
            current_model = model_match.group(1)
        
        if f1_train_match and current_model:
            f1_train_score = float(f1_train_match.group(1))
            f1_train_scores.append((current_model, f1_train_score))
        
        if f1_val_match and current_model:
            f1_val_score = float(f1_val_match.group(1))
            f1_val_scores.append((current_model, f1_val_score))
    
    return f1_train_scores, f1_val_scores

#splits list into dict with individual models
def separate_scores_by_model(f1_train_scores, f1_val_scores):
    model_train_scores = defaultdict(list)
    model_val_scores = defaultdict(list)

    for model, score in f1_train_scores:
        model_train_scores[model].append(score)

    for model, score in f1_val_scores:
        model_val_scores[model].append(score)

    return model_train_scores, model_val_scores



#example how to use
if __name__ == "__main__":
    log_file_path = 'logs/2024-07-23__23-06-14/log.log'
    f1_train_scores, f1_val_scores = parse_f1_scores(log_file_path)    
    model_train_scores, model_val_scores = separate_scores_by_model(f1_train_scores, f1_val_scores)
    
    for model in model_train_scores:
        train_scores = model_train_scores[model]
        val_scores = model_val_scores[model]
        
        epochs = range(1, len(train_scores) + 1)
        
        plt.plot(epochs, train_scores, marker='o', markersize=4, linestyle='-', label=f'{model} - Training')
        plt.plot(epochs, val_scores, marker='x', markersize=4, linestyle='--', label=f'{model} - Validation')

    plt.xlabel('Training Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training vs. Validation F1 Score by Epoch')
    plt.legend()
    plt.grid(True)
    plt.show()
