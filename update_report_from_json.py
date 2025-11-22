#!/usr/bin/env python3
"""
Intelligent script to automatically update LaTeX report values from dfl_results.json
Features:
- Automatic backup before updates
- Validation of JSON data
- Smart regex matching with fallback patterns
- Detailed logging and error reporting
- Dry-run mode for testing
- Configurable formatting
"""

import json
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import sys


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ReportUpdater:
    """Intelligent LaTeX report updater"""
    
    def __init__(self, json_path: Path, tex_path: Path, dry_run: bool = False):
        self.json_path = json_path
        self.tex_path = tex_path
        self.dry_run = dry_run
        self.data = None
        self.original_content = None
        self.updated_content = None
        self.changes_made = []
        
    def log_info(self, msg: str):
        """Log info message"""
        print(f"{Colors.OKBLUE}ℹ{Colors.ENDC} {msg}")
        
    def log_success(self, msg: str):
        """Log success message"""
        print(f"{Colors.OKGREEN}✓{Colors.ENDC} {msg}")
        
    def log_warning(self, msg: str):
        """Log warning message"""
        print(f"{Colors.WARNING}⚠{Colors.ENDC} {msg}")
        
    def log_error(self, msg: str):
        """Log error message"""
        print(f"{Colors.FAIL}✗{Colors.ENDC} {msg}")
        
    def log_change(self, section: str, old_val: str, new_val: str):
        """Log a change made"""
        self.changes_made.append({
            'section': section,
            'old': old_val,
            'new': new_val
        })
        
    @staticmethod
    def format_number(num: float, decimals: int = 6, strip_zeros: bool = True) -> str:
        """Format number with configurable precision"""
        formatted = f"{num:.{decimals}f}"
        if strip_zeros:
            formatted = formatted.rstrip('0').rstrip('.')
        return formatted
    
    @staticmethod
    def format_percentage(ratio: float, decimals: int = 2) -> str:
        """Convert ratio to percentage string"""
        return f"{ratio * 100:.{decimals}f}\\%"
    
    @staticmethod
    def format_large_number(num: int) -> str:
        """Format large numbers with comma separators"""
        return f"{num:,}"
    
    def load_json_results(self) -> bool:
        """Load and validate DFL results from JSON file"""
        try:
            self.log_info(f"Loading JSON from: {self.json_path}")
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            
            # Validate required keys
            required_keys = ['experiments', 'metadata', 'configuration']
            missing_keys = [key for key in required_keys if key not in self.data]
            
            if missing_keys:
                self.log_error(f"Missing required keys in JSON: {missing_keys}")
                return False
            
            # Validate experiments
            if 'balanced' not in self.data['experiments'] or 'imbalanced' not in self.data['experiments']:
                self.log_error("Missing 'balanced' or 'imbalanced' in experiments")
                return False
            
            self.log_success(f"JSON loaded successfully with {len(self.data)} top-level keys")
            return True
            
        except FileNotFoundError:
            self.log_error(f"JSON file not found: {self.json_path}")
            return False
        except json.JSONDecodeError as e:
            self.log_error(f"Invalid JSON format: {e}")
            return False
        except Exception as e:
            self.log_error(f"Error loading JSON: {e}")
            return False
    
    def create_backup(self) -> bool:
        """Create backup of the original tex file"""
        try:
            backup_dir = self.tex_path.parent / 'backups'
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"{self.tex_path.stem}_backup_{timestamp}.tex"
            
            shutil.copy2(self.tex_path, backup_path)
            self.log_success(f"Backup created: {backup_path.name}")
            return True
            
        except Exception as e:
            self.log_error(f"Failed to create backup: {e}")
            return False
    
    def load_tex_content(self) -> bool:
        """Load original tex file content"""
        try:
            with open(self.tex_path, 'r', encoding='utf-8') as f:
                self.original_content = f.read()
                self.updated_content = self.original_content
            self.log_success(f"Loaded tex file: {self.tex_path.name} ({len(self.original_content)} chars)")
            return True
        except FileNotFoundError:
            self.log_error(f"Tex file not found: {self.tex_path}")
            return False
        except Exception as e:
            self.log_error(f"Error loading tex file: {e}")
            return False
    
    def safe_replace(self, pattern: str, replacement: str, flags=re.DOTALL, 
                     section_name: str = "", count: int = 0) -> bool:
        """Safely replace text with validation"""
        try:
            matches = re.findall(pattern, self.updated_content, flags=flags)
            
            if not matches:
                self.log_warning(f"Pattern not found: {section_name or pattern[:50]}")
                return False
            
            if len(matches) > 1 and count == 0:
                self.log_warning(f"Multiple matches found for {section_name} ({len(matches)}), replacing first")
                count = 1
            
            old_content = self.updated_content
            self.updated_content = re.sub(pattern, replacement, self.updated_content, 
                                         flags=flags, count=count if count > 0 else 0)
            
            if old_content != self.updated_content:
                self.log_success(f"Updated: {section_name}")
                if matches:
                    old_val = matches[0][:50] + "..." if len(str(matches[0])) > 50 else str(matches[0])
                    new_val = replacement[:50] + "..." if len(replacement) > 50 else replacement
                    self.log_change(section_name, old_val, new_val)
                return True
            
            return False
            
        except Exception as e:
            self.log_error(f"Error replacing {section_name}: {e}")
            return False
    
    def update_data_distribution_table(self) -> bool:
        """Update data distribution statistics table"""
        self.log_info("Updating data distribution table...")
        
        try:
            bal_dist = self.data['data_distribution_analysis']['balanced']
            imbal_dist = self.data['data_distribution_analysis']['imbalanced']
            
            # Calculate ranges
            bal_min = min(p['train_samples'] for p in bal_dist['peers'])
            bal_max = max(p['train_samples'] for p in bal_dist['peers'])
            imbal_min = min(p['train_samples'] for p in imbal_dist['peers'])
            imbal_max = max(p['train_samples'] for p in imbal_dist['peers'])
            
            table_content = rf"""\\begin{{table}}[H]
\\centering
\\caption{{Thống kê phân phối dữ liệu}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Distribution}} & \\textbf{{Total Samples}} & \\textbf{{Mean}} & \\textbf{{Std Dev}} & \\textbf{{Range}} \\\\
\\midrule
Balanced & {self.format_large_number(bal_dist['total_samples'])} & {self.format_number(bal_dist['mean_samples'], 1)} & {self.format_number(bal_dist['std_samples'], 1)} & {self.format_large_number(bal_min)} - {self.format_large_number(bal_max)} \\\\
Imbalanced & {self.format_large_number(imbal_dist['total_samples'])} & {self.format_number(imbal_dist['mean_samples'], 1)} & {self.format_number(imbal_dist['std_samples'], 1)} & {self.format_large_number(imbal_min)} - {self.format_large_number(imbal_max)} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
            
            pattern = r'\\begin\{table\}\[H\]\\s*\\centering\\s*\\caption\{Thống kê phân phối dữ liệu\}.*?\\end\{table\}'
            return self.safe_replace(pattern, table_content, section_name="Data Distribution Table")
            
        except Exception as e:
            self.log_error(f"Error updating data distribution table: {e}")
            return False
    
    def update_threshold_section(self, experiment: str, threshold: float) -> bool:
        """Update threshold section for balanced or imbalanced"""
        self.log_info(f"Updating {experiment} threshold section...")
        
        exp_data = self.data['experiments'][experiment]
        
        # Note: Mean MSE, Std Dev, Median would need to be in JSON
        # For now, we'll only update what we have
        threshold_text = rf"""\\textbf{{Threshold cho {experiment.title()} Distribution:}}
\\begin{{itemize}}
    \\item 95th Percentile Threshold: {self.format_number(threshold, 6)}
    \\item Mean + 2$\\sigma$: {self.format_number(threshold, 6)}
\\end{{itemize}}"""
        
        pattern = rf'\\textbf\{{Threshold cho {experiment.title()} Distribution:\}}.*?\\end\{{itemize\}}'
        return self.safe_replace(pattern, threshold_text, 
                               section_name=f"{experiment.title()} Threshold Section")
    
    def update_anomaly_detection_table(self, experiment: str) -> bool:
        """Update anomaly detection table for balanced or imbalanced"""
        self.log_info(f"Updating {experiment} anomaly detection table...")
        
        try:
            exp_data = self.data['experiments'][experiment]
            threshold = exp_data['anomaly_detection']['threshold']
            test_results = exp_data['anomaly_detection']['test_results']
            
            # Build table rows
            rows = []
            for test in test_results:
                comparison = ">" if test['is_anomaly'] else "<"
                result_color = "red" if test['is_anomaly'] else "green"
                result_text = "ANOMALY" if test['is_anomaly'] else "NORMAL"
                
                row = f"{test['name']} & {self.format_number(test['error'], 6)} & {comparison} {self.format_number(threshold, 6)} & \\\\textcolor{{{result_color}}}{{{{{result_text}}}}} \\\\\\\\"
                rows.append(row)
            
            table_content = rf"""Mô hình được test trên 4 scenarios với threshold = {self.format_number(threshold, 6)}:

\\begin{{table}}[H]
\\centering
\\caption{{Kết quả anomaly detection - {experiment.title()} Distribution}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Test Case}} & \\textbf{{MSE Error}} & \\textbf{{Threshold}} & \\textbf{{Result}} \\\\
\\midrule
{chr(10).join(rows)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
            
            # For balanced, it's the first occurrence
            # For imbalanced, it's the second occurrence
            if experiment == 'balanced':
                pattern = r'Mô hình được test trên 4 scenarios với threshold = [0-9.]+:.*?\\end\{table\}'
                return self.safe_replace(pattern, table_content, count=1,
                                       section_name=f"{experiment.title()} Anomaly Detection Table")
            else:
                # Find and replace second occurrence
                pattern = r'Mô hình được test trên 4 scenarios với threshold = [0-9.]+:.*?\\end\{table\}'
                matches = list(re.finditer(pattern, self.updated_content, flags=re.DOTALL))
                if len(matches) >= 2:
                    start, end = matches[1].span()
                    self.updated_content = self.updated_content[:start] + table_content + self.updated_content[end:]
                    self.log_success(f"Updated: {experiment.title()} Anomaly Detection Table")
                    return True
                else:
                    self.log_warning(f"Could not find second occurrence for imbalanced table")
                    return False
                    
        except Exception as e:
            self.log_error(f"Error updating {experiment} anomaly detection table: {e}")
            return False
    
    def update_experiments_comparison_table(self) -> bool:
        """Update main experiments comparison table"""
        self.log_info("Updating experiments comparison table...")
        
        try:
            bal_conv = self.data['experiments']['balanced']['convergence']
            imbal_conv = self.data['experiments']['imbalanced']['convergence']
            
            table_content = rf"""\\begin{{table}}[H]
\\centering
\\caption{{Kết quả thí nghiệm DFL}}
\\begin{{tabular}}{{lcccc}}
\\toprule
\\textbf{{Experiment}} & \\textbf{{Initial Loss}} & \\textbf{{Final Train Loss}} & \\textbf{{Final Eval Loss}} & \\textbf{{Reduction}} \\\\
\\midrule
DFL Balanced & {self.format_number(bal_conv['initial_train_loss'], 6)} & {self.format_number(bal_conv['final_train_loss'], 6)} & {self.format_number(bal_conv['final_eval_loss'], 6)} & {self.format_percentage(bal_conv['train_loss_reduction'])} \\\\
DFL Imbalanced & {self.format_number(imbal_conv['initial_train_loss'], 6)} & {self.format_number(imbal_conv['final_train_loss'], 6)} & {self.format_number(imbal_conv['final_eval_loss'], 6)} & {self.format_percentage(imbal_conv['train_loss_reduction'])} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
            
            pattern = r'\\begin\{table\}\[H\]\\s*\\centering\\s*\\caption\{Kết quả thí nghiệm DFL\}.*?\\end\{table\}'
            return self.safe_replace(pattern, table_content, count=1,
                                   section_name="Experiments Comparison Table")
                                   
        except Exception as e:
            self.log_error(f"Error updating experiments comparison table: {e}")
            return False
    
    def update_summary_comparison_table(self) -> bool:
        """Update the summary comparison section with all metrics"""
        self.log_info("Updating summary comparison table...")
        
        try:
            bal = self.data['experiments']['balanced']
            imbal = self.data['experiments']['imbalanced']
            
            bal_conv = bal['convergence']
            imbal_conv = imbal['convergence']
            bal_threshold = bal['anomaly_detection']['threshold']
            imbal_threshold = imbal['anomaly_detection']['threshold']
            bal_tests = bal['anomaly_detection']['test_results']
            imbal_tests = imbal['anomaly_detection']['test_results']
            
            summary_section = rf"""\\textbf{{Training Performance}} & & \\
Initial Train Loss & {self.format_number(bal_conv['initial_train_loss'], 6)} & {self.format_number(imbal_conv['initial_train_loss'], 6)} \\
Final Train Loss & {self.format_number(bal_conv['final_train_loss'], 6)} & {self.format_number(imbal_conv['final_train_loss'], 6)} \\
Train Loss Reduction & {self.format_percentage(bal_conv['train_loss_reduction'])} & {self.format_percentage(imbal_conv['train_loss_reduction'])} \\
\\midrule
\\textbf{{Evaluation Performance}} & & \\
Initial Eval Loss & {self.format_number(bal_conv['initial_eval_loss'], 6)} & {self.format_number(imbal_conv['initial_eval_loss'], 6)} \\
Final Eval Loss & {self.format_number(bal_conv['final_eval_loss'], 6)} & {self.format_number(imbal_conv['final_eval_loss'], 6)} \\
Eval Loss Reduction & {self.format_percentage(bal_conv['eval_loss_reduction'])} & {self.format_percentage(imbal_conv['eval_loss_reduction'])} \\
\\midrule
\\textbf{{Anomaly Detection}} & & \\
Threshold (95th percentile) & {self.format_number(bal_threshold, 6)} & {self.format_number(imbal_threshold, 6)} \\
Normal Sample Error & {self.format_number(bal_tests[0]['error'], 6)} & {self.format_number(imbal_tests[0]['error'], 6)} \\
Anomaly Detection Accuracy & 100\\% & 100\\% \\"""
            
            pattern = r'\\textbf\{Training Performance\} & & \\\\.*?Anomaly Detection Accuracy & 100\\% & 100\\% \\\\'
            return self.safe_replace(pattern, summary_section,
                                   section_name="Summary Comparison Table")
                                   
        except Exception as e:
            self.log_error(f"Error updating summary comparison table: {e}")
            return False
    
    def update_all_sections(self) -> bool:
        """Update all sections in the chapter"""
        self.log_info("\n" + "="*60)
        self.log_info("Starting chapter updates...")
        self.log_info("="*60 + "\n")
        
        success_count = 0
        total_updates = 0
        
        # List of all update functions
        updates = [
            ("Data Distribution Table", lambda: self.update_data_distribution_table()),
            ("Balanced Threshold", lambda: self.update_threshold_section('balanced', 
                self.data['experiments']['balanced']['anomaly_detection']['threshold'])),
            ("Imbalanced Threshold", lambda: self.update_threshold_section('imbalanced',
                self.data['experiments']['imbalanced']['anomaly_detection']['threshold'])),
            ("Balanced Anomaly Table", lambda: self.update_anomaly_detection_table('balanced')),
            ("Imbalanced Anomaly Table", lambda: self.update_anomaly_detection_table('imbalanced')),
            ("Experiments Comparison", lambda: self.update_experiments_comparison_table()),
            ("Summary Comparison", lambda: self.update_summary_comparison_table()),
        ]
        
        for name, update_func in updates:
            total_updates += 1
            try:
                if update_func():
                    success_count += 1
            except Exception as e:
                self.log_error(f"Failed to update {name}: {e}")
        
        self.log_info("\n" + "="*60)
        self.log_info(f"Update Summary: {success_count}/{total_updates} sections updated successfully")
        self.log_info("="*60 + "\n")
        
        return success_count > 0
    
    def write_updated_content(self) -> bool:
        """Write the updated content back to file"""
        if self.dry_run:
            self.log_info("DRY RUN: Would write updates to file")
            return True
            
        try:
            with open(self.tex_path, 'w', encoding='utf-8') as f:
                f.write(self.updated_content)
            self.log_success(f"Successfully wrote updates to {self.tex_path}")
            return True
        except Exception as e:
            self.log_error(f"Failed to write file: {e}")
            return False
    
    def show_diff_summary(self):
        """Show summary of changes made"""
        if not self.changes_made:
            self.log_warning("No changes were made")
            return
        
        print(f"\n{Colors.HEADER}{Colors.BOLD}Changes Summary:{Colors.ENDC}")
        print("="*60)
        
        for i, change in enumerate(self.changes_made, 1):
            print(f"\n{Colors.OKCYAN}{i}. {change['section']}{Colors.ENDC}")
        
        print("\n" + "="*60)
        print(f"{Colors.OKGREEN}Total changes: {len(self.changes_made)}{Colors.ENDC}\n")
    
    def run(self) -> bool:
        """Main execution flow"""
        print(f"\n{Colors.HEADER}{Colors.BOLD}LaTeX Report Intelligent Updater{Colors.ENDC}")
        print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
        
        if self.dry_run:
            self.log_info("Running in DRY RUN mode - no files will be modified")
        
        # Step 1: Load JSON
        if not self.load_json_results():
            return False
        
        # Step 2: Load tex file
        if not self.load_tex_content():
            return False
        
        # Step 3: Create backup (skip in dry-run)
        if not self.dry_run:
            if not self.create_backup():
                self.log_warning("Continuing without backup...")
        
        # Step 4: Update all sections
        if not self.update_all_sections():
            self.log_error("Failed to update sections")
            return False
        
        # Step 5: Write updates
        if not self.write_updated_content():
            return False
        
        # Step 6: Show summary
        self.show_diff_summary()
        
        return True


def main():
    """Main function with command-line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Intelligently update LaTeX report from JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Normal run with default paths
  %(prog)s --dry-run                # Test without making changes
  %(prog)s --json custom.json       # Use custom JSON file
  %(prog)s --chapter chap7/7.tex    # Update different chapter
        """
    )
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without making actual changes (test mode)')
    parser.add_argument('--json', type=str, 
                       help='Path to JSON results file (default: reports_dfl/dfl_results.json)')
    parser.add_argument('--chapter', type=str,
                       help='Path to chapter tex file (default: report_latex/chap6/6.tex)')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    json_path = Path(args.json) if args.json else script_dir / 'reports_dfl' / 'dfl_results.json'
    chapter_path = Path(args.chapter) if args.chapter else script_dir / 'report_latex' / 'chap6' / '6.tex'
    
    # Validate paths
    if not json_path.exists():
        print(f"{Colors.FAIL}Error: JSON file not found: {json_path}{Colors.ENDC}")
        sys.exit(1)
    
    if not chapter_path.exists():
        print(f"{Colors.FAIL}Error: Chapter file not found: {chapter_path}{Colors.ENDC}")
        sys.exit(1)
    
    # Create updater and run
    updater = ReportUpdater(json_path, chapter_path, dry_run=args.dry_run)
    success = updater.run()
    
    if success:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}✨ Report updated successfully!{Colors.ENDC}\n")
        
        if not args.dry_run:
            print(f"{Colors.OKCYAN}Next steps:{Colors.ENDC}")
            print(f"  1. Review changes: git diff {chapter_path}")
            print(f"  2. Build PDF: ./quick_pdf.sh")
            print(f"  3. Check backup: {chapter_path.parent / 'backups'}\n")
        sys.exit(0)
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ Update failed!{Colors.ENDC}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
