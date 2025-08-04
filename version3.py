"""
Fixed CSV Regression Testing Tool
Validates data types using schema instead of exact value matching
Enhanced with file extension validation and timestamp pattern matching
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import sys
import re


class ValidationResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during comparison"""
    issue_type: str
    severity: ValidationResult
    message: str
    row_index: Optional[int] = None
    column: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None


@dataclass
class ValidationReport:
    """Complete validation report for a file comparison"""
    file_name: str
    reference_file: str
    generated_file: str
    timestamp: datetime = field(default_factory=datetime.now)
    overall_result: ValidationResult = ValidationResult.PASS
    issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report"""
        self.issues.append(issue)
        if issue.severity == ValidationResult.FAIL:
            self.overall_result = ValidationResult.FAIL
        elif issue.severity == ValidationResult.WARNING and self.overall_result == ValidationResult.PASS:
            self.overall_result = ValidationResult.WARNING
    
    def generate_summary(self):
        """Generate summary statistics"""
        self.summary = {
            'total_issues': len(self.issues),
            'failures': sum(1 for issue in self.issues if issue.severity == ValidationResult.FAIL),
            'warnings': sum(1 for issue in self.issues if issue.severity == ValidationResult.WARNING),
            'passes': sum(1 for issue in self.issues if issue.severity == ValidationResult.PASS)
        }


class DataTypeValidator:
    """Handles data type validation and formatting checks"""
    
    @staticmethod
    def validate_date_format(value: Any, column_name: str) -> Tuple[bool, str]:
        """Validate date format - supports multiple common formats"""
        if pd.isna(value) or value == '':
            return True, ""
        
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
            r'^\d{4}/\d{2}/\d{2}$',  # YYYY/MM/DD
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',  # YYYY-MM-DDTHH:MM:SS (ISO 8601)
        ]
        
        str_value = str(value).strip()
        for pattern in date_patterns:
            if re.match(pattern, str_value):
                return True, ""
        
        return False, f"Invalid date format: {value}"
    
    @staticmethod
    def validate_numeric_format(value: Any, data_type: str) -> Tuple[bool, str]:
        """Validate numeric format (int/float)"""
        if pd.isna(value) or value == '':
            return True, ""
        
        try:
            if data_type == 'int':
                # Check if it's a valid integer (could be stored as float like 123.0)
                float_val = float(value)
                if float_val.is_integer():
                    return True, ""
                else:
                    return False, f"Expected integer but got: {value}"
            elif data_type == 'float':
                float(value)
                return True, ""
            return True, ""
        except (ValueError, TypeError):
            return False, f"Invalid {data_type} format: {value}"
    
    @staticmethod
    def validate_string_format(value: Any) -> Tuple[bool, str]:
        """Validate string format"""
        if pd.isna(value):
            return True, ""
        # String validation can be extended based on requirements
        return True, ""
    
    @staticmethod
    def get_inferred_type(value: Any) -> str:
        """Infer the data type of a value"""
        if pd.isna(value) or value == '':
            return 'null'
        
        # Try to infer type
        str_value = str(value).strip()
        
        # Check if it's a date
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',
            r'^\d{2}/\d{2}/\d{4}$',
            r'^\d{2}-\d{2}-\d{4}$',
            r'^\d{4}/\d{2}/\d{2}$',
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, str_value):
                return 'date'
        
        # Check if it's numeric
        try:
            float_val = float(value)
            if float_val.is_integer():
                return 'int'
            else:
                return 'float'
        except (ValueError, TypeError):
            return 'string'


class CSVComparator:
    """Core comparison logic for CSV files"""
    
    def __init__(self, schema_config: Dict[str, Any]):
        self.schema_config = schema_config
        self.validator = DataTypeValidator()
        self.logger = logging.getLogger(__name__)
    
    def compare_headers(self, ref_df: pd.DataFrame, gen_df: pd.DataFrame, 
                       file_name: str) -> List[ValidationIssue]:
        """Compare column headers between reference and generated files"""
        issues = []
        expected_columns = self.schema_config[file_name]['columns']
        
        # Check if all expected columns exist
        missing_columns = set(expected_columns) - set(gen_df.columns)
        for col in missing_columns:
            issues.append(ValidationIssue(
                issue_type="missing_column",
                severity=ValidationResult.FAIL,
                message=f"Missing expected column: {col}",
                column=col
            ))
        
        # Check for unexpected columns
        unexpected_columns = set(gen_df.columns) - set(expected_columns)
        for col in unexpected_columns:
            issues.append(ValidationIssue(
                issue_type="unexpected_column",
                severity=ValidationResult.WARNING,
                message=f"Unexpected column found: {col}",
                column=col
            ))
        
        # Check column order
        common_columns = [col for col in expected_columns if col in gen_df.columns]
        if common_columns != gen_df.columns[:len(common_columns)].tolist():
            issues.append(ValidationIssue(
                issue_type="column_order",
                severity=ValidationResult.WARNING,
                message="Column order differs from expected",
                expected_value=expected_columns,
                actual_value=gen_df.columns.tolist()
            ))
        
        return issues
    
    def compare_data_types(self, gen_df: pd.DataFrame, file_name: str) -> List[ValidationIssue]:
        """Validate data types and formats - This is the main validation now"""
        issues = []
        type_mapping = self.schema_config[file_name]['types']
        
        for column, expected_type in type_mapping.items():
            if column not in gen_df.columns:
                continue
                
            for idx, value in gen_df[column].items():
                # Skip empty values for optional validation
                if pd.isna(value) or value == '':
                    continue
                
                # Validate based on expected type
                if 'date' in column.lower() or expected_type == 'date':
                    is_valid, error_msg = self.validator.validate_date_format(value, column)
                elif expected_type in ['int', 'float']:
                    is_valid, error_msg = self.validator.validate_numeric_format(value, expected_type)
                else:
                    is_valid, error_msg = self.validator.validate_string_format(value)
                
                if not is_valid:
                    issues.append(ValidationIssue(
                        issue_type="data_type_validation",
                        severity=ValidationResult.FAIL,
                        message=f"Column '{column}': {error_msg}",
                        row_index=idx,
                        column=column,
                        actual_value=value,
                        expected_value=f"Valid {expected_type}"
                    ))
        
        return issues
    
    def compare_structure_and_completeness(self, ref_df: pd.DataFrame, gen_df: pd.DataFrame, 
                                         file_name: str) -> List[ValidationIssue]:
        """Compare structure and completeness without exact value matching"""
        issues = []
        
        # Get row count comparison settings from schema
        row_count_config = self.schema_config[file_name].get('row_count_policy', {})
        allow_more_rows = row_count_config.get('allow_more_rows', True)
        allow_fewer_rows = row_count_config.get('allow_fewer_rows', False)
        max_row_difference = row_count_config.get('max_row_difference', None)
        
        ref_row_count = len(ref_df)
        gen_row_count = len(gen_df)
        row_difference = gen_row_count - ref_row_count
        
        # Handle row count differences based on policy
        if row_difference > 0:  # More rows in generated file
            if not allow_more_rows:
                issues.append(ValidationIssue(
                    issue_type="row_count_mismatch",
                    severity=ValidationResult.FAIL,
                    message=f"Generated file has more rows than reference: {gen_row_count} vs {ref_row_count}",
                    expected_value=ref_row_count,
                    actual_value=gen_row_count
                ))
            elif max_row_difference and row_difference > max_row_difference:
                issues.append(ValidationIssue(
                    issue_type="row_count_excessive",
                    severity=ValidationResult.WARNING,
                    message=f"Generated file has {row_difference} more rows than reference (max allowed: {max_row_difference})",
                    expected_value=f"<= {ref_row_count + max_row_difference}",
                    actual_value=gen_row_count
                ))
            else:
                # Just log as informational
                issues.append(ValidationIssue(
                    issue_type="row_count_increase",
                    severity=ValidationResult.PASS,
                    message=f"Generated file has {row_difference} more rows than reference ({gen_row_count} vs {ref_row_count})",
                    expected_value=ref_row_count,
                    actual_value=gen_row_count
                ))
        
        elif row_difference < 0:  # Fewer rows in generated file
            if not allow_fewer_rows:
                issues.append(ValidationIssue(
                    issue_type="row_count_mismatch",
                    severity=ValidationResult.FAIL,
                    message=f"Generated file has fewer rows than reference: {gen_row_count} vs {ref_row_count}",
                    expected_value=ref_row_count,
                    actual_value=gen_row_count
                ))
            elif max_row_difference and abs(row_difference) > max_row_difference:
                issues.append(ValidationIssue(
                    issue_type="row_count_excessive",
                    severity=ValidationResult.WARNING,
                    message=f"Generated file has {abs(row_difference)} fewer rows than reference (max allowed: {max_row_difference})",
                    expected_value=f">= {ref_row_count - max_row_difference}",
                    actual_value=gen_row_count
                ))
            else:
                issues.append(ValidationIssue(
                    issue_type="row_count_decrease",
                    severity=ValidationResult.WARNING,
                    message=f"Generated file has {abs(row_difference)} fewer rows than reference ({gen_row_count} vs {ref_row_count})",
                    expected_value=ref_row_count,
                    actual_value=gen_row_count
                ))
        
        return issues
    
    def validate_key_uniqueness(self, gen_df: pd.DataFrame, file_name: str) -> List[ValidationIssue]:
        """Validate that key columns have unique combinations"""
        issues = []
        key_columns = self.schema_config[file_name].get('key_columns', [])
        
        if not key_columns:
            return issues
        
        # Check if all key columns exist
        missing_keys = set(key_columns) - set(gen_df.columns)
        if missing_keys:
            issues.append(ValidationIssue(
                issue_type="missing_key_columns",
                severity=ValidationResult.FAIL,
                message=f"Missing key columns: {list(missing_keys)}",
                expected_value=key_columns,
                actual_value=list(set(key_columns) - missing_keys)
            ))
            return issues
        
        # Check for duplicate key combinations
        key_combinations = gen_df[key_columns]
        duplicates = key_combinations[key_combinations.duplicated(keep=False)]
        
        if not duplicates.empty:
            duplicate_rows = duplicates.index.tolist()
            issues.append(ValidationIssue(
                issue_type="duplicate_keys",
                severity=ValidationResult.FAIL,
                message=f"Found duplicate key combinations in rows: {duplicate_rows}",
                column=', '.join(key_columns),
                actual_value=len(duplicates)
            ))
        
        return issues
    
    def check_null_values(self, gen_df: pd.DataFrame, file_name: str) -> List[ValidationIssue]:
        """Check for missing/null values in required columns"""
        issues = []
        
        # Get required columns from schema
        required_columns = self.schema_config[file_name].get('required_columns', [])
        
        for col in required_columns:
            if col in gen_df.columns:
                null_count = gen_df[col].isna().sum()
                if null_count > 0:
                    null_indices = gen_df[gen_df[col].isna()].index.tolist()
                    issues.append(ValidationIssue(
                        issue_type="null_values",
                        severity=ValidationResult.WARNING,
                        message=f"Found {null_count} null values in required column '{col}' at rows: {null_indices[:10]}{'...' if len(null_indices) > 10 else ''}",
                        column=col,
                        actual_value=null_count
                    ))
        
        return issues


class CSVRegressionTester:
    """Main class for orchestrating CSV regression testing"""
    
    def __init__(self, schema_file: str, reference_dir: str, generated_dir: str):
        self.schema_file = Path(schema_file)
        self.reference_dir = Path(reference_dir)
        self.generated_dir = Path(generated_dir)
        self.schema_config = self._load_schema()
        self.comparator = CSVComparator(self.schema_config)
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('csv_regression_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load schema configuration from JSON file"""
        try:
            with open(self.schema_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load schema file: {e}")
    
    def _extract_base_name_from_timestamped_file(self, filename: str) -> str:
        """
        Extract base name from timestamped filename
        Examples:
        - CashRecov_20250702184047.csv -> CashRecov
        - cashrecov_20250702184047.xlsx -> cashrecov
        - SomeFile_20250101120000.csv -> SomeFile
        """
        # Remove file extension first
        base_name = Path(filename).stem
        
        # Pattern to match timestamp suffixes like _20250702184047
        timestamp_pattern = r'_\d{14}$'  # Matches _YYYYMMDDHHMMSS at end
        
        # Remove timestamp suffix if present
        base_name = re.sub(timestamp_pattern, '', base_name)
        
        return base_name
    
    def _find_reference_file(self, base_name: str) -> Optional[Path]:
        """Find reference file with case-insensitive matching and any supported extension"""
        supported_extensions = ['.csv', '.xlsx', '.xls']
        
        # Try exact match first
        for ext in supported_extensions:
            ref_file = self.reference_dir / f"{base_name}{ext}"
            if ref_file.exists():
                return ref_file
        
        # Try case-insensitive match
        for ext in supported_extensions:
            for file_path in self.reference_dir.glob(f"*{ext}"):
                file_base = file_path.stem
                if file_base.lower() == base_name.lower():
                    return file_path
        
        return None
    
    def _find_generated_file(self, base_name: str) -> Optional[Path]:
        """Find generated file with pattern matching for timestamped files"""
        supported_extensions = ['.csv', '.xlsx', '.xls']
        
        # Try exact match first
        for ext in supported_extensions:
            gen_file = self.generated_dir / f"{base_name}{ext}"
            if gen_file.exists():
                return gen_file
        
        # Try pattern matching for timestamped files
        for ext in supported_extensions:
            # Look for files that start with base_name followed by timestamp pattern
            pattern = f"{base_name}_*{ext}"
            for file_path in self.generated_dir.glob(pattern):
                # Verify it matches the timestamp pattern
                extracted_base = self._extract_base_name_from_timestamped_file(file_path.name)
                if extracted_base.lower() == base_name.lower():
                    return file_path
        
        # Try case-insensitive pattern matching
        for ext in supported_extensions:
            for file_path in self.generated_dir.glob(f"*{ext}"):
                extracted_base = self._extract_base_name_from_timestamped_file(file_path.name)
                if extracted_base.lower() == base_name.lower():
                    return file_path
        
        return None
    
    def _validate_file_extensions(self, ref_file: Path, gen_file: Path) -> Optional[ValidationIssue]:
        """Validate that file extensions match between reference and generated files"""
        ref_ext = ref_file.suffix.lower()
        gen_ext = gen_file.suffix.lower()
        
        if ref_ext != gen_ext:
            return ValidationIssue(
                issue_type="file_extension_mismatch",
                severity=ValidationResult.WARNING,
                message=f"File extension mismatch: Reference file has '{ref_ext}' but generated file has '{gen_ext}'",
                expected_value=ref_ext,
                actual_value=gen_ext
            )
        
        return None
    
    def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load CSV file with error handling"""
        try:
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'cp1252']
            separators = [',', ';', '\t']
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                        if len(df.columns) > 1:  # Basic validation
                            return df
                    except:
                        continue
            
            # If Excel file, try reading as Excel
            if file_path.suffix in ['.xlsx', '.xls']:
                return pd.read_excel(file_path)
            
            raise ValueError("Could not read file with any encoding/separator combination")
            
        except Exception as e:
            raise ValueError(f"Failed to load file {file_path}: {e}")
    
    def test_file(self, file_name: str) -> ValidationReport:
        """Test a single CSV file against its reference - ENHANCED WITH TIMESTAMP PATTERN MATCHING"""
        self.logger.info(f"Testing file: {file_name}")
        
        # Extract base name without extension and timestamp
        base_name = self._extract_base_name_from_timestamped_file(file_name)
        
        # Initialize report
        report = ValidationReport(
            file_name=file_name,
            reference_file="",
            generated_file=""
        )
        
        try:
            # Find reference and generated files using pattern matching
            ref_file = self._find_reference_file(base_name)
            gen_file = self._find_generated_file(base_name)
            
            # Update report with actual file paths
            report.reference_file = str(ref_file) if ref_file else f"NOT FOUND: {base_name}.*"
            report.generated_file = str(gen_file) if gen_file else f"NOT FOUND: {base_name}.*"
            
            if not ref_file:
                report.add_issue(ValidationIssue(
                    issue_type="reference_file_not_found",
                    severity=ValidationResult.FAIL,
                    message=f"Reference file not found for base name: {base_name} (searched extensions: .csv, .xlsx, .xls with case-insensitive matching)"
                ))
                return report
            
            if not gen_file:
                report.add_issue(ValidationIssue(
                    issue_type="generated_file_not_found",
                    severity=ValidationResult.FAIL,
                    message=f"Generated file not found for base name: {base_name} (searched with timestamp pattern matching and extensions: .csv, .xlsx, .xls)"
                ))
                return report
            
            # Log successful file matching
            self.logger.info(f"Successfully matched files:")
            self.logger.info(f"  Reference: {ref_file}")
            self.logger.info(f"  Generated: {gen_file}")
            
            # Check file extension compatibility
            extension_issue = self._validate_file_extensions(ref_file, gen_file)
            if extension_issue:
                report.add_issue(extension_issue)
            
            # Load files
            ref_df = self._load_csv_file(ref_file)
            gen_df = self._load_csv_file(gen_file)
            
            # Run validations
            # 1. Check headers/structure
            header_issues = self.comparator.compare_headers(ref_df, gen_df, file_name)
            for issue in header_issues:
                report.add_issue(issue)
            
            # 2. Validate data types (MAIN FOCUS - NOT EXACT VALUES)
            data_type_issues = self.comparator.compare_data_types(gen_df, file_name)
            for issue in data_type_issues:
                report.add_issue(issue)
            
            # 3. Check structure and completeness (row counts, etc.)
            structure_issues = self.comparator.compare_structure_and_completeness(ref_df, gen_df, file_name)
            for issue in structure_issues:
                report.add_issue(issue)
            
            # 4. Validate key uniqueness
            key_issues = self.comparator.validate_key_uniqueness(gen_df, file_name)
            for issue in key_issues:
                report.add_issue(issue)
            
            # 5. Check for null values in required columns
            null_issues = self.comparator.check_null_values(gen_df, file_name)
            for issue in null_issues:
                report.add_issue(issue)
            
        except Exception as e:
            report.add_issue(ValidationIssue(
                issue_type="processing_error",
                severity=ValidationResult.FAIL,
                message=f"Error processing file: {str(e)}"
            ))
        
        report.generate_summary()
        return report
    
    def run_all_tests(self) -> List[ValidationReport]:
        """Run tests for all configured files"""
        reports = []
        
        for file_name in self.schema_config.keys():
            report = self.test_file(file_name)
            reports.append(report)
        
        return reports
    
    def generate_html_report(self, reports: List[ValidationReport], output_file: str = "regression_report.html"):
        """Generate HTML report"""
        html_content = self._build_html_report(reports)
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {output_file}")
    
    def _build_html_report(self, reports: List[ValidationReport]) -> str:
        """Build HTML report content"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CSV Type Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .pass { color: green; }
                .fail { color: red; }
                .warning { color: orange; }
                .file-report { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }
                .issue { margin: 5px 0; padding: 5px; background-color: #f9f9f9; }
                .issue.pass { border-left: 4px solid green; }
                .issue.warning { border-left: 4px solid orange; }
                .issue.fail { border-left: 4px solid red; }
                .extension-mismatch { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .info { background-color: #e6f3ff; }
                .highlight { background-color: #ffffcc; padding: 10px; border-radius: 5px; margin-bottom: 15px; }
                .file-info { background-color: #f8f9fa; padding: 8px; margin: 5px 0; border-radius: 3px; font-size: 0.9em; }
            </style>
        </head>
        <body>
            <h1>CSV Type Validation Report</h1>
            <div class="highlight">
                <strong>Note:</strong> This report focuses on data type validation according to schema, 
                not exact value matching. Files are validated for structure, data types, completeness, and file extension compatibility.
            </div>
            <div class="summary">
                <h2>Summary</h2>
        """
        
        total_files = len(reports)
        passed = sum(1 for r in reports if r.overall_result == ValidationResult.PASS)
        failed = sum(1 for r in reports if r.overall_result == ValidationResult.FAIL)
        warnings = sum(1 for r in reports if r.overall_result == ValidationResult.WARNING)
        
        # Count extension mismatches
        extension_mismatches = sum(1 for r in reports 
                                 for issue in r.issues 
                                 if issue.issue_type == "file_extension_mismatch")
        
        html += f"""
                <p>Total Files: {total_files}</p>
                <p class="pass">Passed: {passed}</p>
                <p class="fail">Failed: {failed}</p>
                <p class="warning">Warnings: {warnings}</p>
                <p class="warning">Extension Mismatches: {extension_mismatches}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for report in reports:
            status_class = report.overall_result.value.lower()
            html += f"""
            <div class="file-report">
                <h2 class="{status_class}">{report.file_name} - {report.overall_result.value}</h2>
                <div class="file-info">
                    <strong>Reference:</strong> {report.reference_file}<br>
                    <strong>Generated:</strong> {report.generated_file}
                </div>
                <p>Issues Found: {len(report.issues)}</p>
            """
            
            if report.issues:
                html += "<h3>Issues:</h3>"
                for issue in report.issues:
                    issue_class = issue.severity.value.lower()
                    extra_class = ""
                    if issue.issue_type == "file_extension_mismatch":
                        extra_class = " extension-mismatch"
                    
                    html += f"""
                    <div class="issue {issue_class}{extra_class}">
                        <strong>{issue.issue_type}</strong>: {issue.message}
                    """
                    
                    if issue.expected_value and issue.actual_value:
                        html += f"""
                        <br><small>Expected: {issue.expected_value} | Actual: {issue.actual_value}</small>
                        """
                    
                    html += "</div>"
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main execution function"""
    # Configuration
    schema_file = "schema.json"
    reference_dir = "reference_files"
    generated_dir = "generated_files"
    
    try:
        # Initialize tester
        tester = CSVRegressionTester(schema_file, reference_dir, generated_dir)
        
        # Run tests
        reports = tester.run_all_tests()
        
        # Generate reports
        tester.generate_html_report(reports)
        
        # Print summary
        print("\n" + "="*60)
        print("CSV TYPE VALIDATION SUMMARY")
        print("="*60)
        print("Focus: Data types, structure, and file extension validation")
        print("="*60)
        
        extension_mismatches = 0
        for report in reports:
            status = "✓" if report.overall_result == ValidationResult.PASS else "✗"
            
            # Check for extension mismatches
            has_extension_mismatch = any(issue.issue_type == "file_extension_mismatch" 
                                       for issue in report.issues)
            if has_extension_mismatch:
                extension_mismatches += 1
                extension_indicator = " [EXT MISMATCH]"
            else:
                extension_indicator = ""
            
            print(f"{status} {report.file_name}: {report.overall_result.value} ({len(report.issues)} issues){extension_indicator}")
        
        print("="*60)
        print(f"Total files tested: {len(reports)}")
        print(f"Passed: {sum(1 for r in reports if r.overall_result == ValidationResult.PASS)}")
        print(f"Failed: {sum(1 for r in reports if r.overall_result == ValidationResult.FAIL)}")
        print(f"Warnings: {sum(1 for r in reports if r.overall_result == ValidationResult.WARNING)}")
        print(f"Extension mismatches: {extension_mismatches}")
        
    except Exception as e:
        print(f"Error running regression tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()