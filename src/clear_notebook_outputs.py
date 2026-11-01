"""Clear Notebook outputs before sharing; does not upload or commit anything."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', type=Path)
    args = parser.parse_args()
    with args.notebook.open(encoding='utf-8') as notebook_file:
        notebook = json.load(notebook_file)
    if not isinstance(notebook, dict) or not isinstance(notebook.get('cells'), list):
        parser.error('notebook must contain a cells list')
    for cell in notebook['cells']:
        if not isinstance(cell, dict):
            parser.error('notebook contains an invalid cell')
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    with args.notebook.open('w', encoding='utf-8') as notebook_file:
        json.dump(notebook, notebook_file, ensure_ascii=False, indent=1)
        notebook_file.write('\n')
    print(f'Cleared saved outputs: {args.notebook}')


if __name__ == '__main__':
    main()
