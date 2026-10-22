"""Executa o notebook final do início ao fim numa pasta temporária vazia.

Executar da raiz: python -m atividade4.validar_notebook
A pasta temporária não contém o repositório: se o notebook depender de
caminhos locais ou de módulos fora dos embutidos, a execução falha.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from atividade4.criar_notebook import NOME, TARGETS

root = Path(__file__).resolve().parents[1]
base = root.parent / '.jupyter'
kernel = base / 'kernels/atividade4'
kernel.mkdir(parents=True, exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'Atividade 4', 'language': 'python'}), encoding='utf-8')
os.environ['JUPYTER_PATH'] = str(base)
os.environ.pop('PYTHONPATH', None)

source = TARGETS[0]
with tempfile.TemporaryDirectory(prefix='atividade4_limpo_') as folder:
    work = Path(folder) / NOME
    shutil.copy(source, work)
    notebook = nbformat.read(work, as_version=4)
    NotebookClient(notebook, timeout=900, kernel_name='atividade4',
                   resources={'metadata': {'path': folder}}).execute()
    nbformat.validate(notebook)
    errors = [o for c in notebook.cells if c.cell_type == 'code'
              for o in c.get('outputs', []) if o.get('output_type') == 'error']
    if errors:
        raise RuntimeError(f'{len(errors)} saída(s) de erro no notebook.')
    for target in TARGETS:
        nbformat.write(notebook, target)
    print('Executado em pasta limpa:', folder)
print(f'{NOME}: {sum(c.cell_type == "code" for c in notebook.cells)} células de código, sem erros.')
