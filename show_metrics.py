import os
import json
import pandas as pd

with open('settings.json') as f:
    s = json.load(f)

base = s['base_path']
model_dir = s['model_output_dir']
model_name = s['model_output_name']
results_csv = os.path.join(base, model_dir, model_name, 'results.csv')
output_file = os.path.join(base, 'metrics.txt')

if not os.path.exists(results_csv):
    print(f"Error: {results_csv} not found")
    exit(1)

df = pd.read_csv(results_csv)
last = df.iloc[-1]
best_idx = df['metrics/mAP50(B)'].idxmax()
best = df.iloc[best_idx]

with open(output_file, 'w') as f:
    f.write("Training Metrics\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Final Epoch: {int(last['epoch'])}\n\n")
    f.write(f"Precision: {last['metrics/precision(B)']:.4f} ({last['metrics/precision(B)']*100:.2f}%)\n")
    f.write(f"Recall:    {last['metrics/recall(B)']:.4f} ({last['metrics/recall(B)']*100:.2f}%)\n")
    f.write(f"mAP50:     {last['metrics/mAP50(B)']:.4f} ({last['metrics/mAP50(B)']*100:.2f}%)\n")
    f.write(f"mAP50-95:  {last['metrics/mAP50-95(B)']:.4f} ({last['metrics/mAP50-95(B)']*100:.2f}%)\n\n")
    
    if best_idx != len(df) - 1:
        f.write(f"Best Model (epoch {int(best['epoch'])}):\n")
        f.write(f"mAP50:     {best['metrics/mAP50(B)']:.4f} ({best['metrics/mAP50(B)']*100:.2f}%)\n")
        f.write(f"Precision: {best['metrics/precision(B)']:.4f} ({best['metrics/precision(B)']*100:.2f}%)\n")
        f.write(f"Recall:    {best['metrics/recall(B)']:.4f} ({best['metrics/recall(B)']*100:.2f}%)\n")

print(f"Metrics saved to {output_file}")
