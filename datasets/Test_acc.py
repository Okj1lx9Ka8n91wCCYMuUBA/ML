import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
matches_path = '/home/dark/Documents/GitHub/ML/matches_with_scores.csv'
grants_path = '/home/dark/Documents/GitHub/ML/grants.csv'

try:
    matches_df = pd.read_csv(matches_path)
    grants_df = pd.read_csv(grants_path)
    
    if 'true_labels' in matches_df.columns and 'predicted_labels' in matches_df.columns:
        true_labels = matches_df['true_labels']
        predicted_labels = matches_df['predicted_labels']
        
        # Calculate metrics
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        roc_auc = roc_auc_score(true_labels, matches_df['score']) if 'score' in matches_df.columns else None
        average_precision = average_precision_score(true_labels, matches_df['score']) if 'score' in matches_df.columns else None
        
        metrics_results = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Average Precision': average_precision
        }
        
        metrics_df = pd.DataFrame(metrics_results.items(), columns=['Metric', 'Value'])
        metrics_df.to_csv('/mnt/data/evaluated_metrics.csv', index=False)
        print("Metrics have been calculated and saved: evaluated_metrics.csv")
    else:
        print("The matches_with_scores.csv file does not contain the required columns (true_labels, predicted_labels).")
except FileNotFoundError as e:
    print(f"Error: {e}")

