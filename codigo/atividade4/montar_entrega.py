"""Empacota a entrega revisada: notebook, apresentação, parecer e código de reprodução.

Executar da raiz: python -m atividade4.montar_entrega
"""
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
REVISAO = ROOT.parent / 'saida'
target = REVISAO / 'Atividade_4_Entrega_Revisada.zip'
files = []
for folder in ['mtsa', 'atividade4']:
    for path in (ROOT/folder).rglob('*'):
        if (path.is_file() and '__pycache__' not in path.parts
                and 'validacao_pdf' not in path.parts):
            files.append((path, 'codigo/' + path.relative_to(ROOT).as_posix()))
for relative in ['examples/MTSA.ipynb', 'examples/sample_data_utils.py',
                 'examples/linear_ts_with_confounder_n5000_vars6_lag2_gaussian.csv',
                 'tests/test_atividade4.py', 'requirements-atividade4.txt']:
    files.append((ROOT/relative, 'codigo/' + relative))
for name in ['Luan_Cesar_de_Souza_Nascimento_Atividade_4.ipynb',
             'Atividade_4_Apresentacao_Revisada.pdf', 'PARECER_REVISAO.md']:
    files.append((REVISAO/name, name))
files.append((ROOT/'atividade4/README.md', 'LEIA-ME.md'))
manifest = {}
with ZipFile(target, 'w', compression=ZIP_DEFLATED) as archive:
    for path, relative in sorted(files, key=lambda item: item[1]):
        content = path.read_bytes()
        archive.writestr('Atividade_4_Entrega/' + relative, content)
        manifest[relative] = hashlib.sha256(content).hexdigest()
    archive.writestr('Atividade_4_Entrega/MANIFESTO_SHA256.json', json.dumps(manifest, indent=2))
with ZipFile(target) as archive:
    if archive.testzip() is not None:
        raise RuntimeError('Falha na integridade do ZIP.')
    for relative, expected in manifest.items():
        if hashlib.sha256(archive.read('Atividade_4_Entrega/'+relative)).hexdigest() != expected:
            raise RuntimeError(f'Hash divergente: {relative}')
print(f'{target}: {len(manifest)} arquivos, {target.stat().st_size:,} bytes. Integridade conferida.')
