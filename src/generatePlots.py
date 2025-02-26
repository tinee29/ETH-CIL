import matplotlib.pyplot as plt
import sys
sys.path.append('.')

import src.logParser 

savePlots= True #will overwrite existing plots
saveLocation = 'results/plots'
f1_log_location = 'logs/2024-07-27__14-18-44/log.log'


def plot_f1_scores(log_file_path):
    f1_train_scores, f1_val_scores = src.logParser.parse_f1_scores(log_file_path)
    model_train_scores, model_val_scores = src.logParser.separate_scores_by_model(f1_train_scores, f1_val_scores)
    plt.figure(figsize=(10, 5))  
    for model, scores in model_train_scores.items():
        epochs = range(1, len(scores) + 1)
        plt.plot(epochs, scores, marker='o', markersize=0, linestyle='-', label=f'{model} - Training')
    plt.xlabel('Training Epoch')
    plt.ylabel('F1 Score')
    #plt.title('Training F1 Score by Epoch')#no title for report
    plt.legend()
    plt.grid(True)
    if savePlots:
        plt.savefig(f'results/plots/epoch_vs_f1training_combined.png')
    plt.show()

    plt.figure(figsize=(10, 5))  # Optional: Adjust figure size
    for model, scores in model_val_scores.items():
        epochs = range(1, len(scores) + 1)
        plt.plot(epochs, scores, marker='x', markersize=0, linestyle='-', label=f'{model} - Validation')
    plt.xlabel('Training Epoch')
    plt.ylabel('F1 Score')
    #plt.title('Validation F1 Score by Epoch') #no title for report
    plt.legend()
    plt.grid(True)
    if savePlots:
        plt.savefig(f'results/plots/epoch_vs_f1validation_combined.png')
    plt.show()


#our submission training run crash while straining street paint twice, so the last dataset is assembled 2 seperate logs
def plot_f1_scores_crashworkaround(log_file_paths, savePlots=False):
    combined_train_scores = {}
    combined_val_scores = {}

    for log_file_path in log_file_paths:
        f1_train_scores, f1_val_scores = src.logParser.parse_f1_scores(log_file_path)
        model_train_scores, model_val_scores = src.logParser.separate_scores_by_model(f1_train_scores, f1_val_scores)

        for model, scores in model_train_scores.items():
            if model in combined_train_scores:
                combined_train_scores[model].extend(scores)
            else:
                combined_train_scores[model] = scores

        for model, scores in model_val_scores.items():
            if model in combined_val_scores:
                combined_val_scores[model].extend(scores)
            else:
                combined_val_scores[model] = scores

    # Remove the second-to-last dataset from combined scores (this is the crashed street paint training run that was restarted)
    train_models = list(combined_train_scores.keys())
    val_models = list(combined_val_scores.keys())

    if len(train_models) > 1:
        second_to_last_train_model = train_models[-2]
        del combined_train_scores[second_to_last_train_model]

    if len(val_models) > 1:
        second_to_last_val_model = val_models[-2]
        del combined_val_scores[second_to_last_val_model]

    # Rename the last dataset to "street_paint" (name with all models is very long)
    if train_models:
        last_train_model = train_models[-1]
        combined_train_scores["street_paint"] = combined_train_scores.pop(last_train_model)

    if val_models:
        last_val_model = val_models[-1]
        combined_val_scores["street_paint"] = combined_val_scores.pop(last_val_model)

    plt.figure(figsize=(10, 5))  
    for model, scores in combined_train_scores.items():
        epochs = range(1, len(scores) + 1)
        plt.plot(epochs, scores, marker='o', markersize=0, linestyle='-', label=f'{model} - Training')
    
    plt.xlabel('Training Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    if savePlots:
        plt.savefig('results/plots/epoch_vs_f1training_combined.png')
    plt.show()

    plt.figure(figsize=(10, 5)) 
    for model, scores in combined_val_scores.items():
        epochs = range(1, len(scores) + 1)
        plt.plot(epochs, scores, marker='x', markersize=0, linestyle='-', label=f'{model} - Validation')
    
    plt.xlabel('Training Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    if savePlots:
        plt.savefig('results/plots/epoch_vs_f1validation_combined.png')
    plt.show()

plot_f1_scores(f1_log_location)

log_file_paths = ['logs/2024-07-27__14-18-44/log.log', 'logs/2024-07-28__14-46-20/log.log', 'logs/2024-07-28__15-56-03/log.log']
plot_f1_scores_crashworkaround(log_file_paths, savePlots=True)
