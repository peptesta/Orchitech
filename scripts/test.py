import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from dotenv import dotenv_values

# --- PROJECT IMPORTS ---
from app.model_fun.inference import loadDevice, loadModel, getValues6ClassModel
from app.model_fun.preprocess_data import getTransforms

# --- CONFIGURATION ---
config = dotenv_values(".env")
BASE_DIR = "/app"
MODELS_ROOT = os.path.join(BASE_DIR, "app/models/detection_models/test")
CACHE_DIR = os.path.join(BASE_DIR, "test_results/cache")

DATASETS = {
    "test": {
        "input": os.path.join(BASE_DIR, "datasets/test"), 
        "output": os.path.join(BASE_DIR, "test_results/test")
    }
}

WIDTH, HEIGHT = 256, 512
CLASS_NAMES = ['O. exaltata', 'O. garganica', 'O. incubacea', 'O. majellensis', 'O. sphegodes', 'O. sphegodes_Palena']
MEAN = [float(x) for x in config.get("MEAN", '0.541428 0.539673 0.352925').split()]
STD = [float(x) for x in config.get("STD", '0.210250 0.231360 0.199286').split()]

def save_detailed_metrics(y_true, y_pred, y_smart, model_name, model_type, output_dir):
    """Generates detailed performance metrics and comparison plots."""
    metrics = {}
    for mode, labels in [("Standard", y_pred), ("SmartCrop", y_smart)]:
        acc = accuracy_score(y_true, labels)
        p, r, f1, _ = precision_recall_fscore_support(y_true, labels, average='macro', zero_division=0)
        metrics[mode] = {"Accuracy": acc, "Precision": p, "Recall": r, "F1": f1}

        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix(y_true, labels, labels=CLASS_NAMES), 
                    annot=True, fmt='d', cmap='Blues' if mode == "Standard" else 'Greens', 
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"{model_name} ({model_type})\nMode: {mode} | Acc: {acc:.2%}")
        plt.savefig(os.path.join(output_dir, f"CM_{model_name}_{mode}.png"))
        plt.close()

    # Comparison Plot: Accuracy, Precision, Recall, F1
    df_plot = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Mode'})
    df_melted = df_plot.melt(id_vars='Mode', var_name='Metric', value_name='Score')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Mode', palette='muted')
    plt.ylim(0, 1.1)
    plt.title(f"Performance Metrics Comparison: {model_name} ({model_type})")
    for p in plt.gca().patches:
        plt.gca().annotate(f'{p.get_height():.2%}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=9, color='black', xytext=(0, 7), 
                           textcoords='offset points')
    plt.savefig(os.path.join(output_dir, f"METRICS_{model_name}.png"))
    plt.close()
    
    return metrics

def run_multi_model_benchmark():
    device = loadDevice()
    transform_pipeline = getTransforms(WIDTH, HEIGHT, True, MEAN, STD)
    
    # Discovery of models in subfolders
    model_tasks = []
    for subfolder in ['smart', 'nosmart']:
        folder_path = os.path.join(MODELS_ROOT, subfolder)
        if os.path.exists(folder_path):
            for f in os.listdir(folder_path):
                if f.endswith('.pt'):
                    model_tasks.append({
                        "name": os.path.splitext(f)[0],
                        "path": os.path.join(folder_path, f),
                        "type": subfolder
                    })

    if not model_tasks:
        print("No models found in smart/ or nosmart/ subfolders.")
        return

    for ds_name, ds_paths in DATASETS.items():
        output_path = ds_paths["output"]
        os.makedirs(output_path, exist_ok=True)

        # PHASE 2: LOADING CACHED DATA
        ds_cache_dir = os.path.join(CACHE_DIR, ds_name)
        cached_files = [os.path.join(ds_cache_dir, f) for f in os.listdir(ds_cache_dir) if f.endswith('.pt')]
        
        # PHASE 3: MULTI-MODEL INFERENCE
        summary_results = []

        for task in model_tasks:
            print(f"\n--- Testing {task['type'].upper()} model: {task['name']} ---")
            model = loadModel(task['path'], len(CLASS_NAMES), device)
            
            y_true, y_pred, y_smart_pred = [], [], []

            with torch.no_grad():
                for pt_file in tqdm(cached_files, desc=f"Inference: {task['name']}"):
                    data = torch.load(pt_file)
                    t_std, t_crop = data["tensor_std"].to(device), data["tensor_crop"].to(device)
                    
                    p_idx, _, _ = getValues6ClassModel(model, t_std, device)
                    ps_idx, _, _ = getValues6ClassModel(model, t_crop, device)
                    
                    y_true.append(data["label"])
                    y_pred.append(CLASS_NAMES[p_idx])
                    y_smart_pred.append(CLASS_NAMES[ps_idx])

            # Save results and get metrics
            results = save_detailed_metrics(y_true, y_pred, y_smart_pred, task['name'], task['type'], output_path)
            
            summary_results.append({
                "Model": task['name'],
                "Category": task['type'],
                "Std_Acc": results["Standard"]["Accuracy"],
                "Smart_Acc": results["SmartCrop"]["Accuracy"],
                "Delta": results["SmartCrop"]["Accuracy"] - results["Standard"]["Accuracy"],
                "Smart_F1": results["SmartCrop"]["F1"]
            })

        # Final Summary Table
        summary_df = pd.DataFrame(summary_results).sort_values(by="Smart_Acc", ascending=False)
        print(f"\n--- Benchmark Summary for {ds_name} ---")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(os.path.join(output_path, "global_benchmark_summary.csv"), index=False)

        # Global Comparison Chart (Model Category Performance)
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=summary_df, x="Std_Acc", y="Smart_Acc", hue="Category", style="Category", s=100)
        plt.plot([0, 1], [0, 1], color='red', linestyle='--') # Diagonal line
        plt.title("Standard vs Smart Crop Accuracy per Model Category")
        plt.xlabel("Accuracy on Original Image")
        plt.ylabel("Accuracy on Smart Crop")
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "global_category_comparison.png"))

if __name__ == "__main__":
    run_multi_model_benchmark()