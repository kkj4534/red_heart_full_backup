#!/usr/bin/env python3
"""
Method inventory extractor for Red Heart project.
Extracts all method definitions from Python files and categorizes them.
"""

import os
import re
import ast
import sys
from typing import Dict, List, Tuple, Set
from pathlib import Path

class MethodInventory:
    def __init__(self):
        self.methods = []
        self.embedding_methods = []
        self.gpu_methods = []
        self.async_methods = []
        self.class_methods = []
        self.static_methods = []
        self.property_methods = []
        
    def extract_methods_from_file(self, file_path: str) -> List[Dict]:
        """Extract all method definitions from a Python file."""
        methods = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Get lines for line number lookup
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    method_info = self._extract_method_info(node, file_path, lines)
                    if method_info:
                        methods.append(method_info)
                        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Fallback to regex-based extraction
            methods.extend(self._regex_extract_methods(file_path))
            
        return methods
    
    def _extract_method_info(self, node, file_path: str, lines: List[str]) -> Dict:
        """Extract detailed information about a method from AST node."""
        method_info = {
            'file': os.path.basename(file_path),
            'full_path': file_path,
            'name': node.name,
            'line_number': node.lineno,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'is_class_method': False,
            'is_static_method': False,
            'is_property': False,
            'parameters': [],
            'decorators': [],
            'docstring': ast.get_docstring(node) or "",
            'category': 'general'
        }
        
        # Extract parameters
        if node.args:
            for arg in node.args.args:
                method_info['parameters'].append(arg.arg)
            
            # Check for *args
            if node.args.vararg:
                method_info['parameters'].append(f"*{node.args.vararg.arg}")
            
            # Check for **kwargs
            if node.args.kwarg:
                method_info['parameters'].append(f"**{node.args.kwarg.arg}")
        
        # Extract decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorator_name = decorator.id
                method_info['decorators'].append(decorator_name)
                
                # Check for special decorators
                if decorator_name == 'classmethod':
                    method_info['is_class_method'] = True
                elif decorator_name == 'staticmethod':
                    method_info['is_static_method'] = True
                elif decorator_name == 'property':
                    method_info['is_property'] = True
                    
        # Create signature
        params_str = ', '.join(method_info['parameters'])
        async_prefix = 'async ' if method_info['is_async'] else ''
        method_info['signature'] = f"{async_prefix}def {node.name}({params_str})"
        
        # Categorize method
        method_info['category'] = self._categorize_method(method_info)
        
        return method_info
    
    def _regex_extract_methods(self, file_path: str) -> List[Dict]:
        """Fallback regex-based method extraction."""
        methods = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                # Match function definitions
                match = re.match(r'^\s*(async\s+)?def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*?)\):', line)
                if match:
                    is_async = match.group(1) is not None
                    method_name = match.group(2)
                    params = match.group(3)
                    
                    method_info = {
                        'file': os.path.basename(file_path),
                        'full_path': file_path,
                        'name': method_name,
                        'line_number': i,
                        'is_async': is_async,
                        'is_class_method': False,
                        'is_static_method': False,
                        'is_property': False,
                        'parameters': [p.strip() for p in params.split(',') if p.strip()],
                        'decorators': [],
                        'docstring': "",
                        'signature': line.strip(),
                        'category': 'general'
                    }
                    
                    # Check for decorators in previous lines
                    for j in range(max(0, i-5), i):
                        if j < len(lines):
                            decorator_match = re.match(r'^\s*@([a-zA-Z_][a-zA-Z0-9_]*)', lines[j])
                            if decorator_match:
                                method_info['decorators'].append(decorator_match.group(1))
                    
                    method_info['category'] = self._categorize_method(method_info)
                    methods.append(method_info)
                    
        except Exception as e:
            print(f"Error in regex extraction for {file_path}: {e}")
            
        return methods
    
    def _categorize_method(self, method_info: Dict) -> str:
        """Categorize method based on name and content."""
        name = method_info['name'].lower()
        file_name = method_info['file'].lower()
        
        # Embedding-related methods
        if any(keyword in name for keyword in ['embed', 'vector', 'semantic', 'similarity', 'encode', 'decode']):
            return 'embedding'
        
        # GPU-related methods
        if any(keyword in name for keyword in ['gpu', 'cuda', 'device', 'tensor', 'torch']):
            return 'gpu'
        
        # Async methods
        if method_info['is_async']:
            return 'async'
        
        # Data processing methods
        if any(keyword in name for keyword in ['load', 'process', 'transform', 'convert', 'parse']):
            return 'data_processing'
        
        # Training methods
        if any(keyword in name for keyword in ['train', 'fit', 'optimize', 'update', 'learn']):
            return 'training'
        
        # Inference methods
        if any(keyword in name for keyword in ['predict', 'infer', 'classify', 'analyze']):
            return 'inference'
        
        # Utility methods
        if any(keyword in name for keyword in ['init', 'setup', 'config', 'validate', 'check']):
            return 'utility'
        
        # File-based categorization
        if any(keyword in file_name for keyword in ['semantic', 'embed']):
            return 'embedding'
        if any(keyword in file_name for keyword in ['gpu', 'cuda']):
            return 'gpu'
        if any(keyword in file_name for keyword in ['async']):
            return 'async'
        
        return 'general'
    
    def scan_directory(self, directory: str) -> None:
        """Scan directory for Python files and extract methods."""
        exclude_patterns = [
            'for_learn_dataset',
            '__pycache__',
            '.git',
            'venv',
            'env',
            '.pytest_cache',
            'node_modules'
        ]
        
        for root, dirs, files in os.walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    methods = self.extract_methods_from_file(file_path)
                    self.methods.extend(methods)
    
    def categorize_methods(self) -> None:
        """Categorize extracted methods."""
        for method in self.methods:
            category = method['category']
            
            if category == 'embedding':
                self.embedding_methods.append(method)
            elif category == 'gpu':
                self.gpu_methods.append(method)
            elif category == 'async':
                self.async_methods.append(method)
                
            if method['is_class_method']:
                self.class_methods.append(method)
            elif method['is_static_method']:
                self.static_methods.append(method)
            elif method['is_property']:
                self.property_methods.append(method)
    
    def generate_inventory_report(self) -> str:
        """Generate a comprehensive inventory report."""
        report = []
        
        # Header
        report.append("# RED HEART PROJECT - METHOD INVENTORY")
        report.append("# Generated automatically - Complete method mapping")
        report.append(f"# Total methods found: {len(self.methods)}")
        report.append("")
        
        # Summary statistics
        report.append("## SUMMARY STATISTICS")
        report.append(f"Total methods: {len(self.methods)}")
        report.append(f"Embedding methods: {len(self.embedding_methods)}")
        report.append(f"GPU methods: {len(self.gpu_methods)}")
        report.append(f"Async methods: {len(self.async_methods)}")
        report.append(f"Class methods: {len(self.class_methods)}")
        report.append(f"Static methods: {len(self.static_methods)}")
        report.append(f"Property methods: {len(self.property_methods)}")
        report.append("")
        
        # File statistics
        files = set(method['file'] for method in self.methods)
        report.append(f"Files with methods: {len(files)}")
        report.append("")
        
        # Embedding methods section
        if self.embedding_methods:
            report.append("## EMBEDDING METHODS")
            for method in sorted(self.embedding_methods, key=lambda x: (x['file'], x['name'])):
                report.append(f"{method['file']}:{method['line_number']} - {method['signature']}")
                if method['docstring']:
                    report.append(f"    # {method['docstring'][:100]}...")
            report.append("")
        
        # GPU methods section
        if self.gpu_methods:
            report.append("## GPU METHODS")
            for method in sorted(self.gpu_methods, key=lambda x: (x['file'], x['name'])):
                report.append(f"{method['file']}:{method['line_number']} - {method['signature']}")
                if method['docstring']:
                    report.append(f"    # {method['docstring'][:100]}...")
            report.append("")
        
        # Async methods section
        if self.async_methods:
            report.append("## ASYNC METHODS")
            for method in sorted(self.async_methods, key=lambda x: (x['file'], x['name'])):
                report.append(f"{method['file']}:{method['line_number']} - {method['signature']}")
                if method['docstring']:
                    report.append(f"    # {method['docstring'][:100]}...")
            report.append("")
        
        # Methods by category
        categories = {}
        for method in self.methods:
            cat = method['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(method)
        
        for category, methods in sorted(categories.items()):
            report.append(f"## {category.upper()} METHODS ({len(methods)})")
            for method in sorted(methods, key=lambda x: (x['file'], x['name'])):
                decorators = f"@{', @'.join(method['decorators'])}" if method['decorators'] else ""
                report.append(f"{method['file']}:{method['line_number']} - {decorators} {method['signature']}")
            report.append("")
        
        # All methods alphabetically
        report.append("## ALL METHODS (ALPHABETICAL)")
        for method in sorted(self.methods, key=lambda x: x['name']):
            async_marker = "[ASYNC]" if method['is_async'] else ""
            class_marker = "[CLASS]" if method['is_class_method'] else ""
            static_marker = "[STATIC]" if method['is_static_method'] else ""
            prop_marker = "[PROPERTY]" if method['is_property'] else ""
            
            markers = " ".join(filter(None, [async_marker, class_marker, static_marker, prop_marker]))
            
            report.append(f"{method['name']} - {method['file']}:{method['line_number']} {markers}")
            report.append(f"    {method['signature']}")
            if method['docstring']:
                report.append(f"    # {method['docstring'][:150]}...")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to run the method inventory extraction."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "/mnt/c/large_project/linux_red_heart"
    
    print(f"Scanning directory: {directory}")
    
    inventory = MethodInventory()
    inventory.scan_directory(directory)
    inventory.categorize_methods()
    
    print(f"Found {len(inventory.methods)} methods")
    
    # Generate report
    report = inventory.generate_inventory_report()
    
    # Save to file
    output_file = os.path.join(directory, "methods_inventory.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Inventory saved to: {output_file}")

if __name__ == "__main__":
    main()