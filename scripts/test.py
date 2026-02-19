#!/usr/bin/env python3
"""
Orchid Classification Testing Script
- Runs inference on all datasets
- Generates comparison graphs and summary reports
"""

import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
API_BASE = "http://localhost:5000"
INFERENCE_API = f"{API_BASE}/inference"

DB_PATH = "/home/giuseppe/Scrivania/Tirocinio/db_immagini"

DATASETS = {
    "cropped": {
        "internet": f"{DB_PATH}/immagini_cropped/immagini_internet",
        "local": f"{DB_PATH}/immagini_cropped/immagini_locali"
    },
    "original": {
        "internet": f"{DB_PATH}/immagini_originali/immagini_internet_originali",
        "local": f"{DB_PATH}/immagini_originali/immagini_locali_originali"
    }
}

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


@dataclass
class Result:
    filename: str
    true_class: str
    predicted_class: str
    confidence: float
    dataset_type: str
    source: str
    correct: bool
    all_probs: Dict[str, float]


class ProgressTracker:
    """Simple progress tracker for terminal output"""
    
    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.current = 0
        self.errors = 0
        self.desc = desc
    
    def update(self, n: int = 1, error: bool = False):
        self.current += n
        if error:
            self.errors += 1
        self._print()
    
    def _print(self):
        pct = (self.current / self.total) * 100 if self.total > 0 else 0
        bar_len = 30
        filled = int(bar_len * self.current / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        
        status = f"{self.desc}: [{bar}] {self.current}/{self.total} ({pct:.1f}%)"
        if self.errors > 0:
            status += f" | Errors: {self.errors}"
        
        print(f"\r{status}", end="", flush=True)
    
    def finish(self):
        print()


def normalize_class_name(name: str) -> str:
    """
    Normalize class name to match model output format.
    Converts 'O  exaltata' (from folder) to 'O. exaltata' (model format).
    """
    # Replace multiple spaces with single space, then replace space after O with dot
    name = ' '.join(name.split())  # Normalize whitespace
    if name.startswith('O '):
        name = 'O.' + name[2:]  # Replace 'O ' with 'O.'
    return name


def discover_images(dataset_type: str, source: str) -> List[Dict]:
    """Find all images in dataset directory"""
    base_path = Path(DATASETS[dataset_type][source])
    images = []
    
    if not base_path.exists():
        print(f"❌ Path not found: {base_path}")
        return images
    
    for class_dir in sorted(base_path.iterdir()):
        if not class_dir.is_dir():
            continue
            
        # Get class name from folder and normalize it
        folder_name = class_dir.name.replace("_Int", "").strip()
        class_name = normalize_class_name(folder_name)
        
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in IMG_EXTENSIONS:
                images.append({
                    "path": img_file,
                    "filename": img_file.name,
                    "true_class": class_name,  # Now in 'O. exaltata' format
                    "dataset_type": dataset_type,
                    "source": source
                })
    
    print(f"  📁 Found {len(images)} images")
    return images


def run_inference(image_path: Path) -> Optional[Dict]:
    """Run inference on single image"""
    try:
        with open(image_path, 'rb') as f:
            resp = requests.post(
                INFERENCE_API,
                files={'image': f},
                data={'model_strategy': 'standard', 'crop_mode': 'integrated'},
                timeout=60
            )
        
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
            
    except Exception:
        return None


def process_dataset(dataset_type: str, source: str) -> List[Result]:
    """Process entire dataset"""
    print(f"\n{'='*70}")
    print(f"📊 PROCESSING: {dataset_type.upper()} - {source.upper()}")
    print(f"{'='*70}")
    
    images = discover_images(dataset_type, source)
    if not images:
        return []
    
    print(f"  🔍 Running inference...")
    results = []
    tracker = ProgressTracker(len(images), "  Inference")
    
    for img in images:
        data = run_inference(img['path'])
        
        if data:
            pred_class = data.get('predicted_class', 'Unknown')
            results.append(Result(
                filename=img['filename'],
                true_class=img['true_class'],
                predicted_class=pred_class,
                confidence=data.get('confidence', 0.0),
                dataset_type=dataset_type,
                source=source,
                correct=pred_class == img['true_class'],
                all_probs=data.get('all_classes_probs', {})
            ))
        else:
            results.append(Result(
                filename=img['filename'],
                true_class=img['true_class'],
                predicted_class='Error',
                confidence=0.0,
                dataset_type=dataset_type,
                source=source,
                correct=False,
                all_probs={}
            ))
            tracker.update(error=True)
        
        tracker.update()
    
    tracker.finish()
    
    correct = sum(1 for r in results if r.correct)
    print(f"  ✅ Results: {correct}/{len(results)} correct ({correct/len(results):.1%})")
    
    return results


def create_visualizations(results: List[Result], output_dir: Path):
    """Generate comparison charts"""
    output_dir.mkdir(exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in results])
    
    if df.empty:
        print("No data to visualize")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {
        'correct': '#27ae60', 'wrong': '#e74c3c',
        'cropped': '#3498db', 'original': '#e67e22',
        'internet': '#9b59b6', 'local': '#f1c40f'
    }
    
    # 1. Dataset Comparison: Cropped vs Original
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dataset Comparison: Cropped vs Original', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax = axes[0, 0]
    ds_stats = df.groupby('dataset_type')['correct'].mean()
    bars = ax.bar(ds_stats.index, ds_stats.values, 
                  color=[colors['cropped'], colors['original']], edgecolor='black')
    ax.set_title('Accuracy by Dataset Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, ds_stats.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.1%}', ha='center', fontweight='bold')
    
    # Prediction counts
    ax = axes[0, 1]
    counts = df.groupby('dataset_type')['correct'].value_counts().unstack()
    counts.plot(kind='bar', stacked=True, ax=ax, 
                color=[colors['wrong'], colors['correct']], edgecolor='black')
    ax.set_title('Prediction Counts', fontsize=12, fontweight='bold')
    ax.legend(['Wrong', 'Correct'])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # Confidence distribution
    ax = axes[1, 0]
    for ds_type in ['cropped', 'original']:
        subset = df[df['dataset_type'] == ds_type]
        ax.hist(subset['confidence'], bins=20, alpha=0.6, label=ds_type, 
                color=colors[ds_type], edgecolor='black')
    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
    ax.set_xlabel('Confidence Score')
    ax.legend()
    
    # Confidence by correctness
    ax = axes[1, 1]
    df.boxplot(column='confidence', by='correct', ax=ax)
    ax.set_title('Confidence by Prediction Correctness', fontsize=12, fontweight='bold')
    ax.set_xlabel('Correct Prediction')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Subset Analysis: Internet vs Local
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Subset Analysis: Internet vs Local', fontsize=16, fontweight='bold')
    
    # Four-way accuracy
    ax = axes[0, 0]
    subset_acc = df.groupby(['dataset_type', 'source'])['correct'].mean()
    x_labels = [f"{ds}\n{src}" for ds, src in subset_acc.index]
    bar_colors = [colors['internet'] if 'internet' in idx else colors['local'] 
                  for idx in subset_acc.index]
    bars = ax.bar(range(len(subset_acc)), subset_acc.values, color=bar_colors, edgecolor='black')
    ax.set_xticks(range(len(subset_acc)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_title('Accuracy by Subset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    for bar, val in zip(bars, subset_acc.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                f'{val:.1%}', ha='center', fontsize=9, fontweight='bold')
    
    # Heatmap
    ax = axes[0, 1]
    pivot = df.pivot_table(values='correct', index='dataset_type', 
                          columns='source', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.1%', cmap='RdYlGn', 
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Accuracy Heatmap', fontsize=12, fontweight='bold')
    
    # Internet detailed
    ax = axes[1, 0]
    internet_df = df[df['source'] == 'internet']
    if not internet_df.empty:
        int_stats = internet_df.groupby('dataset_type')['correct'].mean()
        bars = ax.bar(int_stats.index, int_stats.values,
                      color=[colors['cropped'], colors['original']], edgecolor='black')
        ax.set_title('Internet Images: Cropped vs Original', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, int_stats.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.1%}', ha='center', fontweight='bold')
    
    # Local detailed
    ax = axes[1, 1]
    local_df = df[df['source'] == 'local']
    if not local_df.empty:
        loc_stats = local_df.groupby('dataset_type')['correct'].mean()
        bars = ax.bar(loc_stats.index, loc_stats.values,
                      color=[colors['cropped'], colors['original']], edgecolor='black')
        ax.set_title('Local Images: Cropped vs Original', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.1)
        for bar, val in zip(bars, loc_stats.values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, 
                    f'{val:.1%}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_subset_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class Performance
    fig, ax = plt.subplots(figsize=(12, 6))
    class_perf = df.groupby(['true_class', 'dataset_type'])['correct'].mean().unstack()
    class_perf.plot(kind='bar', ax=ax, width=0.8, 
                    color=[colors['cropped'], colors['original']], edgecolor='black')
    ax.set_title('Per-Class Accuracy by Dataset Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend(title='Dataset Type')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / '03_class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📈 Saved 3 visualization files to {output_dir}")


def generate_report(results: List[Result]) -> str:
    """Generate text summary"""
    df = pd.DataFrame([asdict(r) for r in results])
    
    lines = []
    lines.append("=" * 80)
    lines.append(" ORCHID CLASSIFICATION TEST RESULTS")
    lines.append(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    # Overall stats
    total = len(df)
    correct = df['correct'].sum()
    acc = correct / total if total > 0 else 0
    
    lines.append("📊 OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"  Total Images:      {total}")
    lines.append(f"  Correct:           {int(correct)} ({acc:.2%})")
    lines.append(f"  Wrong:             {total - int(correct)} ({1-acc:.2%})")
    lines.append(f"  Avg Confidence:    {df['confidence'].mean():.4f}")
    lines.append("")
    
    # Dataset comparison
    lines.append("📁 DATASET COMPARISON (Cropped vs Original)")
    lines.append("-" * 40)
    for ds_type, group in df.groupby('dataset_type'):
        acc = group['correct'].mean()
        conf = group['confidence'].mean()
        lines.append(f"  {ds_type.capitalize():12} | Acc: {acc:>6.2%} | Conf: {conf:.4f} | n={len(group):>4}")
    lines.append("")
    
    # Subset comparison
    lines.append("🌐 SUBSET COMPARISON (Internet vs Local)")
    lines.append("-" * 40)
    for (ds_type, source), group in df.groupby(['dataset_type', 'source']):
        acc = group['correct'].mean()
        lines.append(f"  {ds_type:12} | {source:8} | Acc: {acc:>6.2%} | n={len(group):>4}")
    lines.append("")
    
    # Class performance
    lines.append("🎯 PER-CLASS PERFORMANCE")
    lines.append("-" * 40)
    class_stats = df.groupby('true_class').agg({
        'correct': ['count', 'sum', 'mean'],
        'confidence': 'mean'
    }).round(4)
    class_stats.columns = ['Count', 'Correct', 'Accuracy', 'Avg_Conf']
    class_stats = class_stats.sort_values('Accuracy', ascending=False)
    
    lines.append(f"  {'Class':<25} {'Count':>6} {'Correct':>8} {'Accuracy':>10} {'Avg Conf':>10}")
    lines.append("  " + "-" * 60)
    for class_name, row in class_stats.iterrows():
        lines.append(f"  {class_name:<25} {int(row['Count']):>6} {int(row['Correct']):>8} "
                    f"{row['Accuracy']:>9.2%} {row['Avg_Conf']:>10.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    with open("./results/summary_report.txt", 'w') as f:
        f.write(report)
    
    return report


def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print(" 🌸 ORCHID CLASSIFICATION TESTING SUITE")
    print("=" * 70)
    print(f" API Base:    {API_BASE}")
    print(f" Datasets:    {sum(len(s) for s in DATASETS.values())} sources")
    print("-" * 70)
    
    # Check API
    try:
        resp = requests.get(API_BASE, timeout=5)
        print(f" ✅ API Status:  OK (HTTP {resp.status_code})")
    except Exception as e:
        print(f" ❌ API Error:   {e}")
        return
    
    print("-" * 70)
    
    # Process all datasets
    all_results = []
    for dataset_type in DATASETS:
        for source in DATASETS[dataset_type]:
            results = process_dataset(dataset_type, source)
            all_results.extend(results)
    
    if not all_results:
        print("\n ❌ No results generated.")
        return
    
    # Final summary
    print("\n" + "=" * 70)
    print(" ✅ PROCESSING COMPLETE")
    print("=" * 70)
    print(f" Total Inferences: {len(all_results)}")
    
    # Generate outputs
    print("\n 📊 Generating visualizations and reports...")
    create_visualizations(all_results, Path("./results"))
    report = generate_report(all_results)
    
    # Save raw data
    pd.DataFrame([asdict(r) for r in all_results]).to_csv("./results/raw_results.csv", index=False)
    
    print("\n" + report)
    print("\n 💾 Files saved:")
    print("    - results/01_dataset_comparison.png")
    print("    - results/02_subset_analysis.png") 
    print("    - results/03_class_performance.png")
    print("    - results/summary_report.txt")
    print("    - results/raw_results.csv")


if __name__ == "__main__":
    main()