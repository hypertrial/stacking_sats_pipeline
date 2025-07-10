import json
import hashlib


def load_notebook(filepath):
    """Load a Jupyter notebook from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_python_cell_indices(notebook):
    """Get the absolute indices of all Python/code cells in the notebook."""
    cells = notebook.get('cells', [])
    python_indices = []
    
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code':
            python_indices.append(i)
    
    return python_indices


def extract_python_cell_source_by_python_index(notebook, python_cell_number):
    """
    Extract the source code from a Python cell by its Python cell number (1-based).
    python_cell_number=1 means the first Python cell, python_cell_number=2 means the second, etc.
    """
    python_indices = get_python_cell_indices(notebook)
    
    # Convert 1-based python cell number to 0-based index in python_indices list
    if python_cell_number < 1 or python_cell_number > len(python_indices):
        return None
    
    absolute_index = python_indices[python_cell_number - 1]
    cells = notebook.get('cells', [])
    cell = cells[absolute_index]
    
    source = cell.get('source', [])
    if isinstance(source, list):
        return ''.join(source)
    return source


def compute_hash(content):
    """Compute SHA256 hash of content."""
    if content is None:
        return None
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def check_cells_identical(template_path, test_path, python_cell_numbers_to_check, expected_python_cell_count=3):
    """
    Check if specified Python cells in test notebook are identical to template notebook.
    Also validates that both notebooks have the expected number of Python cells.
    Only flags differences without attempting to fix them.
    
    Args:
        template_path (str): Path to the template notebook
        test_path (str): Path to the test notebook to be checked
        python_cell_numbers_to_check (list): List of Python cell numbers to check (1-based)
        expected_python_cell_count (int): Expected number of Python cells in both notebooks
    
    Returns:
        dict: Results of the comparison
    """
    # Load both notebooks
    template_nb = load_notebook(template_path)
    test_nb = load_notebook(test_path)
    
    # Get Python cell indices for both notebooks
    template_python_indices = get_python_cell_indices(template_nb)
    test_python_indices = get_python_cell_indices(test_nb)
    
    template_python_count = len(template_python_indices)
    test_python_count = len(test_python_indices)
    
    results = {
        'python_cells_checked': python_cell_numbers_to_check,
        'differences_found': [],
        'all_identical': True,
        'cell_details': {},
        'template_python_count': template_python_count,
        'test_python_count': test_python_count,
        'expected_python_count': expected_python_cell_count,
        'python_count_valid': True,
        'validation_errors': []
    }
    
    print(f"Template notebook has {template_python_count} Python cells")
    print(f"Test notebook has {test_python_count} Python cells")
    print(f"Expected: {expected_python_cell_count} Python cells")
    
    # Validate Python cell count
    if template_python_count != expected_python_cell_count:
        results['python_count_valid'] = False
        results['all_identical'] = False
        error_msg = f"Template notebook has {template_python_count} Python cells, expected {expected_python_cell_count}"
        results['validation_errors'].append(error_msg)
        print(f"❌ {error_msg}")
    
    if test_python_count != expected_python_cell_count:
        results['python_count_valid'] = False
        results['all_identical'] = False
        error_msg = f"Test notebook has {test_python_count} Python cells, expected {expected_python_cell_count}"
        results['validation_errors'].append(error_msg)
        print(f"❌ {error_msg}")
    
    if results['python_count_valid']:
        print("✅ Both notebooks have the correct number of Python cells")
    
    print()
    
    # Check individual cells only if we have the right number of cells
    if results['python_count_valid']:
        for python_cell_num in python_cell_numbers_to_check:
            # Extract cell content from both notebooks
            template_source = extract_python_cell_source_by_python_index(template_nb, python_cell_num)
            test_source = extract_python_cell_source_by_python_index(test_nb, python_cell_num)
            
            # Compute hashes
            template_hash = compute_hash(template_source)
            test_hash = compute_hash(test_source)
            
            cell_result = {
                'template_hash': template_hash,
                'test_hash': test_hash,
                'identical': template_hash == test_hash,
                'template_exists': template_source is not None,
                'test_exists': test_source is not None
            }
            
            results['cell_details'][python_cell_num] = cell_result
            
            print(f"Python Cell {python_cell_num}:")
            print(f"  Template hash: {template_hash}")
            print(f"  Test hash:     {test_hash}")
            
            if template_hash != test_hash:
                results['all_identical'] = False
                results['differences_found'].append(python_cell_num)
                print(f"  ❌ Python Cell {python_cell_num} differs between notebooks")
                
                # Show more details about the difference
                if template_source is None:
                    print(f"     Template Python cell {python_cell_num} not found")
                elif test_source is None:
                    print(f"     Test Python cell {python_cell_num} not found")
                else:
                    print(f"     Both cells exist but content differs")
            else:
                print(f"  ✅ Python Cell {python_cell_num} is identical")
            print()
    else:
        print("⚠️  Skipping individual cell comparison due to incorrect Python cell count")
        print()
    
    return results


def main():
    """Main function to check notebook cell synchronization."""
    template_path = "models/model_development_template.ipynb"
    test_path = "models/model_development_template_copy.ipynb"
    python_cells_to_check = [1, 3]  # Python cell 1 and Python cell 3 (1-based numbering)
    expected_python_cell_count = 3  # Both notebooks should have exactly 3 Python cells
    
    print("🔍 Checking notebook Python cell synchronization...")
    print(f"Template: {template_path}")
    print(f"Test:     {test_path}")
    print(f"Python cells to check: {python_cells_to_check} (1-based numbering)")
    print(f"Expected Python cell count: {expected_python_cell_count}")
    print("=" * 60)
    
    results = check_cells_identical(template_path, test_path, python_cells_to_check, expected_python_cell_count)
    
    print("=" * 60)
    print("📊 SUMMARY:")
    print(f"Python cells checked: {results['python_cells_checked']}")
    print(f"Expected Python cell count: {results['expected_python_count']}")
    print(f"Template Python cells: {results['template_python_count']}")
    print(f"Test Python cells: {results['test_python_count']}")
    print(f"Python cell count valid: {results['python_count_valid']}")
    print(f"Content differences found: {len(results['differences_found'])}")
    print(f"All checks passed: {results['all_identical']}")
    
    if results['all_identical']:
        print("🎉 SUCCESS: All validations passed!")
        print("   - Both notebooks have exactly 3 Python cells")
        print("   - All specified Python cells are strictly identical")
    else:
        print("❌ VALIDATION FAILURES:")
        
        # Report validation errors (cell count issues)
        if results['validation_errors']:
            for error in results['validation_errors']:
                print(f"   - {error}")
        
        # Report content differences
        if results['differences_found']:
            for python_cell_num in results['differences_found']:
                print(f"   - Python Cell {python_cell_num} content differs between notebooks")
        
        print("\nManual review and correction required.")


if __name__ == "__main__":
    main()
