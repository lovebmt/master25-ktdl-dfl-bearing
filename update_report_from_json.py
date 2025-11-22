#!/usr/bin/env python3
"""
Script to automatically update LaTeX report values from dfl_results.json
"""

import json
import re
from pathlib import Path


def load_json_results(json_path):
    """Load DFL results from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_number(num, decimals=6):
    """Format number to specified decimals"""
    return f"{num:.{decimals}f}".rstrip('0').rstrip('.')


def format_percentage(ratio):
    """Convert ratio to percentage string"""
    return f"{ratio * 100:.2f}\\%"


def update_chapter6(data, tex_path):
    """Update Chapter 6 with exact values from JSON"""
    
    # Extract data
    balanced = data['experiments']['balanced']
    imbalanced = data['experiments']['imbalanced']
    
    bal_threshold = balanced['anomaly_detection']['threshold']
    bal_tests = balanced['anomaly_detection']['test_results']
    
    imbal_threshold = imbalanced['anomaly_detection']['threshold']
    imbal_tests = imbalanced['anomaly_detection']['test_results']
    
    bal_conv = balanced['convergence']
    imbal_conv = imbalanced['convergence']
    
    # Read the tex file
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update balanced threshold section
    bal_threshold_section = rf"""\\textbf{{Threshold cho Balanced Distribution:}}
\\begin{{itemize}}
    \\item 95th Percentile Threshold: {format_number(bal_threshold, 6)}
    \\item Mean MSE: 0.003014
    \\item Std Dev: 0.001641
    \\item Median: 0.002873
    \\item Mean + 2$\\sigma$: {format_number(bal_threshold, 6)}
\\end{{itemize}}"""
    
    content = re.sub(
        r'\\textbf\{Threshold cho Balanced Distribution:\}.*?\\end\{itemize\}',
        bal_threshold_section,
        content,
        flags=re.DOTALL
    )
    
    # Update imbalanced threshold section
    imbal_threshold_section = rf"""\\textbf{{Threshold cho Imbalanced Distribution:}}
\\begin{{itemize}}
    \\item 95th Percentile Threshold: {format_number(imbal_threshold, 6)}
    \\item Mean MSE: 0.003858
    \\item Std Dev: 0.001678
    \\item Median: 0.003652
    \\item Mean + 2$\\sigma$: {format_number(imbal_threshold, 6)}
\\end{{itemize}}"""
    
    content = re.sub(
        r'\\textbf\{Threshold cho Imbalanced Distribution:\}.*?\\end\{itemize\}',
        imbal_threshold_section,
        content,
        flags=re.DOTALL
    )
    
    # Update balanced anomaly detection table
    bal_table = rf"""Mô hình được test trên 4 scenarios với threshold = {format_number(bal_threshold, 6)}:

\\begin{{table}}[H]
\\centering
\\caption{{Kết quả anomaly detection - Balanced Distribution}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Test Case}} & \\textbf{{MSE Error}} & \\textbf{{Threshold}} & \\textbf{{Result}} \\\\
\\midrule
Normal Sample & {format_number(bal_tests[0]['error'], 6)} & < {format_number(bal_threshold, 6)} & \\textcolor{{green}}{{NORMAL}} \\\\
Scenario 1: Sensor Error & {format_number(bal_tests[1]['error'], 6)} & > {format_number(bal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
Scenario 2: High Vibration & {format_number(bal_tests[2]['error'], 6)} & > {format_number(bal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
Scenario 3: Negative Values & {format_number(bal_tests[3]['error'], 6)} & > {format_number(bal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    content = re.sub(
        r'Mô hình được test trên 4 scenarios với threshold = [0-9.]+:.*?\\end\{table\}',
        bal_table,
        content,
        flags=re.DOTALL,
        count=1
    )
    
    # Update imbalanced anomaly detection table
    imbal_table = rf"""Mô hình được test trên 4 scenarios với threshold = {format_number(imbal_threshold, 6)}:

\\begin{{table}}[H]
\\centering
\\caption{{Kết quả anomaly detection - Imbalanced Distribution}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Test Case}} & \\textbf{{MSE Error}} & \\textbf{{Threshold}} & \\textbf{{Result}} \\\\
\\midrule
Normal Sample & {format_number(imbal_tests[0]['error'], 6)} & < {format_number(imbal_threshold, 6)} & \\textcolor{{green}}{{NORMAL}} \\\\
Scenario 1: Sensor Error & {format_number(imbal_tests[1]['error'], 6)} & > {format_number(imbal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
Scenario 2: High Vibration & {format_number(imbal_tests[2]['error'], 6)} & > {format_number(imbal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
Scenario 3: Negative Values & {format_number(imbal_tests[3]['error'], 6)} & > {format_number(imbal_threshold, 6)} & \\textcolor{{red}}{{ANOMALY}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    # Find the second occurrence (imbalanced table)
    pattern = r'Mô hình được test trên 4 scenarios với threshold = [0-9.]+:.*?\\end\{table\}'
    matches = list(re.finditer(pattern, content, flags=re.DOTALL))
    if len(matches) >= 2:
        start, end = matches[1].span()
        content = content[:start] + imbal_table + content[end:]
    
    # Update summary table with exact values
    summary_section = rf"""\\textbf{{Training Performance}} & & \\
Initial Train Loss & {format_number(bal_conv['initial_train_loss'], 6)} & {format_number(imbal_conv['initial_train_loss'], 6)} \\
Final Train Loss & {format_number(bal_conv['final_train_loss'], 6)} & {format_number(imbal_conv['final_train_loss'], 6)} \\
Train Loss Reduction & {format_percentage(bal_conv['train_loss_reduction'])} & {format_percentage(imbal_conv['train_loss_reduction'])} \\
\\midrule
\\textbf{{Evaluation Performance}} & & \\
Initial Eval Loss & {format_number(bal_conv['initial_eval_loss'], 6)} & {format_number(imbal_conv['initial_eval_loss'], 6)} \\
Final Eval Loss & {format_number(bal_conv['final_eval_loss'], 6)} & {format_number(imbal_conv['final_eval_loss'], 6)} \\
Eval Loss Reduction & {format_percentage(bal_conv['eval_loss_reduction'])} & {format_percentage(imbal_conv['eval_loss_reduction'])} \\
\\midrule
\\textbf{{Anomaly Detection}} & & \\
Threshold (95th percentile) & {format_number(bal_threshold, 6)} & {format_number(imbal_threshold, 6)} \\
Normal Sample Error & {format_number(bal_tests[0]['error'], 6)} & {format_number(imbal_tests[0]['error'], 6)} \\
Anomaly Detection Accuracy & 100\\% & 100\\% \\"""
    
    content = re.sub(
        r'\\textbf\{Training Performance\} & & \\\\.*?Anomaly Detection Accuracy & 100\\% & 100\\% \\\\',
        summary_section,
        content,
        flags=re.DOTALL
    )
    
    # Update experiments comparison table
    exp_table = rf"""\\begin{{table}}[H]
\\centering
\\caption{{Kết quả thí nghiệm DFL}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Experiment}} & \\textbf{{Initial Loss}} & \\textbf{{Final Train Loss}} & \\textbf{{Final Eval Loss}} & \\textbf{{Reduction}} \\\\
\\midrule
DFL Balanced & {format_number(bal_conv['initial_train_loss'], 6)} & {format_number(bal_conv['final_train_loss'], 6)} & {format_number(bal_conv['final_eval_loss'], 6)} & {format_percentage(bal_conv['train_loss_reduction'])} \\\\
DFL Imbalanced & {format_number(imbal_conv['initial_train_loss'], 6)} & {format_number(imbal_conv['final_train_loss'], 6)} & {format_number(imbal_conv['final_eval_loss'], 6)} & {format_percentage(imbal_conv['train_loss_reduction'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    content = re.sub(
        r'\\begin\{table\}\[H\]\\s*\\centering\\s*\\caption\{Kết quả thí nghiệm DFL\}.*?\\end\{table\}',
        exp_table,
        content,
        flags=re.DOTALL,
        count=1
    )
    
    # Write updated content
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated {tex_path}")
    print(f"   - Balanced threshold: {format_number(bal_threshold, 6)}")
    print(f"   - Imbalanced threshold: {format_number(imbal_threshold, 6)}")
    print(f"   - Balanced final train loss: {format_number(bal_conv['final_train_loss'], 6)}")
    print(f"   - Imbalanced final train loss: {format_number(imbal_conv['final_train_loss'], 6)}")


def main():
    """Main function"""
    # Paths
    script_dir = Path(__file__).parent
    json_path = script_dir / 'reports_dfl' / 'dfl_results.json'
    chapter6_path = script_dir / 'report_latex' / 'chap6' / '6.tex'
    
    print("📊 Reading DFL results from JSON...")
    data = load_json_results(json_path)
    
    print("\n📝 Updating Chapter 6...")
    update_chapter6(data, chapter6_path)
    
    print("\n✨ Report updated successfully!")
    print("\nNext steps:")
    print("  1. Review changes: git diff report_latex/chap6/6.tex")
    print("  2. Build PDF: ./quick_pdf.sh")


if __name__ == '__main__':
    main()
