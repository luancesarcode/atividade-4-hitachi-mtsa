"""Gera a apresentação curta (HTML 1920x1080 -> PDF) a partir dos resultados medidos.

Executar da raiz: python -m atividade4.gerar_apresentacao
Identidade visual do COMPET Moura usada no Site/SIME: azul #233175, creme #EEE9CD,
amarelo #F4AB1C, Anton nos títulos e Montserrat no texto (fontes OFL em apresentacao/fonts).
A impressão em PDF usa o Chrome ou o Edge em modo headless.
"""
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'atividade4/resultados'
ASSETS = ROOT / 'atividade4/apresentacao'
OUT = ROOT.parent / 'saida'
PDF = OUT / 'Atividade_4_Apresentacao_Revisada.pdf'
REPO = 'github.com/CristoferSilva/mtsa-tutorial-compet'
BROWSERS = [r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe']


def br(value, digits=3):
    return f'{value:.{digits}f}'.replace('.', ',')


def asterisk(size, color='#F4AB1C'):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 100 100" aria-hidden="true">'
            f'<path d="M8 50h84M29 14l42 72M29 86l42-72" stroke="{color}" stroke-width="13" fill="none"/></svg>')


def figure(target):
    """Escore do Hitachi no teste (seed 42), nas cores do COMPET."""
    pred = pd.read_csv(RES / 'predicoes_seed42.csv')
    report = json.loads((RES / 'resultados.json').read_text(encoding='utf-8'))
    threshold = next(r['threshold'] for r in report['runs'] if r['seed'] == 42)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 15, 'text.color': '#022561', 'axes.labelcolor': '#022561', 'xtick.color': '#022561', 'ytick.color': '#022561'})
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
    fig.patch.set_alpha(0)
    ax.set_facecolor('#FFFDF3')
    anomalous = pred['y_true'].to_numpy() == 1
    ax.fill_between(pred['time'], 1e-5, 10, where=anomalous, step='mid', color='#EEE9CD', label='anomalia injetada')
    ax.plot(pred['time'], pred['score_hitachi'], lw=0.8, color='#233175', label='escore Hitachi')
    ax.axhline(threshold, color='#F4AB1C', lw=3, label='limiar (p99 da validação)')
    ax.set(yscale='log', ylim=(1e-4, 10), xlim=(3000, 5000), xlabel='linha do CSV (teste)', ylabel='MSE de reconstrução')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', frameon=False, fontsize=12, ncol=3)
    fig.tight_layout()
    fig.savefig(target, transparent=True)
    plt.close(fig)


def slides(report):
    s, b = report['summary'], report['baseline']
    runs = report['runs']
    fps = [r['fp'] for r in runs]
    st = report['settings']
    head = '''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Atividade 4 · Luan César</title>
<style>
@font-face { font-family: Anton; src: url(fonts/anton-regular.ttf); }
@font-face { font-family: Montserrat; src: url(fonts/montserrat-variable.ttf); font-weight: 100 900; }
@page { size: 1920px 1080px; margin: 0; }
:root { --blue:#233175; --blue-text:#022561; --blue-bright:#0268B4; --yellow:#F4AB1C; --yellow-bright:#FEBA12;
         --cream:#EEE9CD; --paper:#FFFDF3; --ink:#0F0E13; --muted:#586076; }
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: var(--ink); }
body { font-family: Montserrat, sans-serif; color: var(--blue-text); -webkit-print-color-adjust: exact; print-color-adjust: exact; }
section { width: 1920px; height: 1080px; position: relative; overflow: hidden; page-break-after: always; padding: 96px 120px; }
section:last-child { page-break-after: auto; }
.blue { background: var(--blue); color: var(--cream); }
.cream { background: var(--cream); color: var(--blue-text); }
.ink { background: var(--ink); color: var(--cream); }
.kicker { font-weight: 800; letter-spacing: .18em; text-transform: uppercase; font-size: 22px; color: var(--yellow); display:flex; align-items:center; gap:14px; }
.cream .kicker { color: #9a6a00; }
h1, h2 { font-family: Anton, sans-serif; font-weight: 400; line-height: .95; letter-spacing: -.01em; }
h1 { font-size: 230px; color: var(--cream); }
h2 { font-size: 112px; margin: 18px 0 34px; }
.blue h2, .ink h2 { color: var(--cream); }
.cream h2 { color: var(--blue); }
.dot { display:inline-block; width:.2em; height:.2em; background: var(--yellow); margin-left:.03em; }
p.lead { font-size: 34px; line-height: 1.45; max-width: 1180px; }
.card { background: var(--paper); color: var(--blue-text); border-radius: 28px; padding: 40px 44px; box-shadow: 14px 14px 0 var(--yellow); }
.card h3 { font-family: Anton; font-weight:400; font-size: 54px; color: var(--blue); margin: 8px 0 14px; }
.card p, .card li { font-size: 31px; line-height: 1.5; }
.tag { display:inline-block; background: var(--yellow); color: var(--ink); font-weight: 800; font-size: 20px; letter-spacing:.12em; padding: 6px 14px; border-radius: 8px; }
.grid2 { display:grid; grid-template-columns: 1fr 1fr; gap: 56px; }
.stat { background: var(--blue); color: var(--cream); border-radius: 18px; padding: 26px 30px; }
.stat small { display:block; font-weight: 800; letter-spacing:.14em; font-size: 18px; text-transform: uppercase; color: var(--cream); opacity:.85; }
.stat b { font-family: Anton; font-weight: 400; font-size: 88px; color: var(--yellow-bright); line-height: 1.05; }
.stat span { display:block; font-size: 20px; margin-top:4px; }
code, .mono { font-family: Consolas, "Cascadia Mono", monospace; }
.code { font-family: Consolas, "Cascadia Mono", monospace; background: #050A2E; color: var(--cream); border-radius: 20px; padding: 30px 36px; font-size: 26px; line-height: 1.6; border-left: 10px solid var(--yellow); }
.code .bad { color: #FF8C7A; } .code .ok { color: var(--yellow-bright); } .code .c { color: #8C93B8; }
.note { border: 2px dashed currentColor; border-radius: 16px; padding: 18px 24px; font-size: 22px; line-height: 1.45; opacity: .9; }
.footer { position:absolute; left:120px; right:120px; bottom:44px; display:flex; justify-content:space-between; font-size:19px; font-weight:600; opacity:.75; }
.flow { display:grid; grid-template-columns: repeat(5, 1fr); gap: 22px; margin-top: 10px; }
.step { background: var(--paper); color: var(--blue-text); border-radius: 22px; padding: 26px 24px; border-top: 10px solid var(--yellow); }
.step small { display:block; font-weight:800; letter-spacing:.12em; font-size:16px; color:#9a6a00; text-transform:uppercase; }
.step b { display:block; font-size: 29px; margin: 10px 0 8px; color: var(--blue); }
.step span { font-size: 20px; line-height:1.4; color: var(--muted); }
.step em { display:block; font-style:normal; font-size:18px; margin-top:12px; color:var(--muted); border-top:1px solid #d1cfbf; padding-top:10px; }
table.cmp { border-collapse: collapse; font-size: 27px; width: 100%; }
table.cmp th, table.cmp td { text-align:left; padding: 16px 18px; border-bottom: 2px solid rgba(238,233,205,.25); }
table.cmp th { font-size: 19px; letter-spacing:.12em; text-transform:uppercase; color: var(--yellow); }
</style></head><body>'''

    cover = f'''<section class="blue">
  <div style="position:absolute; right:0; top:0; width:660px; height:1080px;">
    <div style="position:absolute; left:0; top:0; width:440px; height:230px; background:var(--yellow);"></div>
    <div style="position:absolute; left:150px; top:30px;">{asterisk(170, '#233175')}</div>
    <div style="position:absolute; left:0; top:230px; width:440px; height:792px; background:var(--ink);"></div>
    <div style="position:absolute; left:440px; top:0; width:220px; height:1080px; background:var(--blue-bright);"></div>
    <div style="position:absolute; left:0; top:1022px; width:440px; height:58px; background:var(--cream);"></div>
  </div>
  <div style="position:absolute; left:120px; top:150px; width:1100px;">
    {asterisk(58)}
    <div class="kicker" style="margin-top:34px;">COMPET Moura · Ciência de dados</div>
    <h1 style="margin-top:26px;">ATIVIDADE 4<span class="dot"></span></h1>
    <p style="font-size:40px; font-weight:700; margin-top:26px;">Correção dos dados e Hitachi no MTSA</p>
    <div style="margin-top:52px; display:flex; align-items:stretch; gap:26px;">
      <div style="width:8px; background:var(--yellow); border-radius:4px;"></div>
      <div>
        <div class="kicker" style="font-size:19px;">Base do trabalho</div>
        <div style="font-size:34px; font-weight:700; margin-top:10px;">CristoferSilva / mtsa-tutorial-compet</div>
        <div style="font-size:22px; margin-top:8px; opacity:.85;">{REPO} · commit {report['provenance']['commit'][:7]}</div>
      </div>
    </div>
    <p style="font-size:26px; margin-top:44px; line-height:1.5;">Luan César de Souza Nascimento<br><span style="opacity:.8;">Engenharia de Controle e Automação · UFPE · 21/10/2026</span></p>
  </div>
</section>'''

    desafio = f'''<section class="cream">
  <div class="kicker">{asterisk(28, '#9a6a00')} O desafio</div>
  <h2>Duas partes, um tutorial.</h2>
  <div class="grid2" style="margin-top:20px;">
    <div class="card"><span class="tag">PARTE I</span><h3>Achar o problema</h3>
      <p>Em <code>get_tabular_sample_data()</code>, a função que monta os dados tabulares com anomalias sintéticas.</p></div>
    <div class="card"><span class="tag">PARTE II</span><h3>Hitachi no framework</h3>
      <p>Treinar e avaliar o autoencoder Hitachi com o mesmo fluxo do exemplo do GANF, no conjunto tabular.</p></div>
  </div>
  <p class="note" style="margin-top:64px; max-width:1500px;">Todo o trabalho foi feito sobre o repositório <b class="mono">{REPO}</b> (commit <span class="mono">a595946</span>). Os dados são o CSV sintético do tutorial: 5.000 linhas, 6 variáveis. Não são dados da Moura.</p>
  <div class="footer"><span>Atividade 4 · COMPET Moura</span><span>02</span></div>
</section>'''

    parte1 = f'''<section class="ink">
  <div class="kicker">{asterisk(28)} Parte I · o erro</div>
  <h2>Uma fatia vazia não avisa.</h2>
  <div style="display:grid; grid-template-columns: 1.25fr 1fr; gap:60px;">
    <div>
      <div class="code"><span class="c"># original: o 1º intervalo usa step como FIM</span><br>
(<span class="bad">start_point or step, step</span>)<br>
<span class="c"># os outros três usam step como DURAÇÃO</span><br>
(2*step, 3*step), (4*step, 5*step), ...<br><br>
<span class="c"># correção</span><br>
<span class="ok">first_start = (step if start_point is None<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;else start_point)</span><br>
<span class="ok">(first_start, first_start + step)</span></div>
      <p style="font-size:25px; line-height:1.5; margin-top:30px; opacity:.9;">O <code>or</code> também descartava um início zero. Agora intervalos vazios ou fora dos dados geram erro.</p>
    </div>
    <div style="display:flex; flex-direction:column; gap:22px;">
      <div class="stat"><small>step=13, início=200</small><b>[200:13]</b> <span>→ 0 linhas alteradas</span></div>
      <div class="stat"><small>step=200, início=150</small><b>50 de 200</b> <span>linhas na 1ª faixa</span></div>
      <div class="stat"><small>anomalias no teste original</small><b>676 → 826</b> <span>após a correção</span></div>
    </div>
  </div>
  <div class="footer"><span>Evidência: seção 3 do notebook</span><span>03</span></div>
</section>'''

    montagem = f'''<section class="cream">
  <div class="kicker">{asterisk(28, '#9a6a00')} Parte I · a montagem do teste</div>
  <h2>Ordem temporal importa.</h2>
  <div class="grid2">
    <div class="card"><span class="tag">ORIGINAL</span><h3>Teste por classe</h3>
      <p><code>vstack([normais, anômalos])</code>: trechos colados que nunca foram vizinhos.</p>
      <ul style="margin:18px 0 0 30px;"><li>11 emendas no teste</li><li>121 de 1.641 janelas de 12 linhas cruzam emendas</li><li>treino com linhas posteriores ao teste</li></ul></div>
    <div class="card"><span class="tag">ENTREGA</span><h3>Blocos cronológicos</h3>
      <p>Linhas 0-2999 normais para treino e validação; 3000-4999 para teste, com anomalias só nele.</p>
      <ul style="margin:18px 0 0 30px;"><li>0 emendas</li><li>treino inteiro antes do teste</li><li>padronização ajustada só no treino</li></ul></div>
  </div>
  <p class="note" style="margin-top:46px;">Não é vazamento de rótulo: nenhuma linha está nos dois conjuntos. É perda da ordem temporal, que afeta modelos com janelas como o GANF. É uma mudança de protocolo, separada da correção do intervalo.</p>
  <div class="footer"><span>Evidência: seção 3 do notebook</span><span>04</span></div>
</section>'''

    parte2 = f'''<section class="blue">
  <div class="kicker">{asterisk(28)} Parte II · integração</div>
  <h2>O mesmo fluxo do GANF.</h2>
  <div class="flow">
    <div class="step"><small>Hipóteses</small><b>Hitachi(6)</b><span>6→64→64→8→64→64→6, já existia no tutorial</span><em>GANF: GANF(...)</em></div>
    <div class="step"><small>Erro</small><b>MSELoss</b><span>entrada = alvo, reconstrução</span><em>GANF: LagrangeLoss</em></div>
    <div class="step"><small>Otimizador</small><b>Adam</b><span>taxa 0,001, lote 64</span><em>GANF: Adam</em></div>
    <div class="step"><small>Aprendizado</small><b>Unsupervised Trainer</b><span>30 épocas, só normais</span><em>GANF: LagrangeTrainer</em></div>
    <div class="step"><small>Predição</small><b>Standard Predictor</b><span>+ HitachiScore: MSE por linha, maior = mais anômalo</span><em>GANF: StandardPredictor</em></div>
  </div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap:48px; margin-top:52px; font-size:26px; line-height:1.5;">
    <p>A peça que faltava era o <b>HitachiScore</b>: o GANF já devolve um escore, o Hitachi devolve a reconstrução. O adaptador calcula o erro e tira o canal extra do preditor: <span class="mono">(1, 1, 6, 1) → (1,)</span>.</p>
    <p>Janela 1: cada linha de seis variáveis é um exemplo. Com seis entradas, a camada de 8 unidades <b>não comprime</b> a entrada. A arquitetura do tutorial foi mantida.</p>
  </div>
  <div class="footer"><span>Evidência: seção 5 do notebook · mtsa/predictors/hitachi_score.py</span><span>05</span></div>
</section>'''

    resultados = f'''<section class="cream">
  <div class="kicker">{asterisk(28, '#9a6a00')} Resultados · 5 seeds</div>
  <h2 style="margin-bottom:26px;">Quase perfeito, e fácil.</h2>
  <div style="display:grid; grid-template-columns: 1fr 1.55fr; gap:50px;">
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; align-content:start;">
      <div class="stat"><small>F1 Hitachi</small><b>{br(s['f1']['mean'])}</b><span>± {br(s['f1']['sd'])} entre seeds</span></div>
      <div class="stat"><small>F1 baseline</small><b>{br(b['f1'])}</b><span>regra sem treino</span></div>
      <div class="stat"><small>Recall</small><b>{br(s['recall']['mean'], 1)}</b><span>nos dois métodos</span></div>
      <div class="stat"><small>Falsos alarmes</small><b>{min(fps)}-{max(fps)}</b><span>de {str(st['test_normals'])[0]}.{str(st['test_normals'])[1:]} normais (baseline: {b['fp']})</span></div>
    </div>
    <div>
      <img src="grafico_seed42.png" style="width:100%; border-radius:20px; background:var(--paper); padding:14px;">
      <p style="font-size:23px; line-height:1.5; margin-top:18px;">As anomalias são degraus de 10 a 46 desvios-padrão. A regra simples também acerta tudo, e a diferença de F1 vem de poucos falsos alarmes. Não dá para dizer que o Hitachi é melhor.</p>
    </div>
  </div>
  <div class="footer"><span>Limiar: percentil 99 da validação normal · seeds 11, 22, 33, 42, 55 fixadas antes</span><span>06</span></div>
</section>'''

    desafiadora = f'''<section class="ink">
  <div class="kicker">{asterisk(28)} Pergunta 1</div>
  <h2>Qual foi a parte mais desafiadora?</h2>
  <p class="lead" style="font-size:40px; max-width:1500px; color:var(--cream);">Confiar nos dados antes de confiar no modelo.</p>
  <div class="grid2" style="margin-top:80px; gap:48px;">
    <div class="card"><p>O erro do intervalo não quebra nada: o código roda e as métricas saem. Ele só aparece quando se conta quantas linhas foram de fato alteradas.</p></div>
    <div class="card"><p>Depois, separar correção de mudança de protocolo, e entender por que até uma regra sem treino tirava nota perto de 1.</p></div>
  </div>
  <div class="footer"><span>Atividade 4 · COMPET Moura</span><span>07</span></div>
</section>'''

    conclusao = f'''<section class="blue">
  <div class="kicker">{asterisk(28)} Pergunta 2</div>
  <h2>Conclusões.</h2>
  <div class="grid2" style="gap:44px;">
    <div class="card"><span class="tag">PARTE I</span><h3>Intervalo corrigido</h3>
      <p>O 1º intervalo usava <code>step</code> como fim. Com fim = início + duração, o teste passa de 676 para 826 anomalias. O teste também deixou de ser montado por classe.</p></div>
    <div class="card"><span class="tag">PARTE II</span><h3>Hitachi integrado</h3>
      <p>Treina e pontua no framework, com F1 {br(s['f1']['mean'])} e recall 1,0. O baseline teve F1 {br(b['f1'])}: o conjunto sintético não diferencia os métodos.</p></div>
  </div>
  <p class="note" style="margin-top:44px; color:var(--cream);">Limites: um arquivo sintético, uma divisão fixa, anomalias grandes e constantes. O GANF não foi reexecutado neste protocolo, então não há comparação entre os dois. Repositório: <span class="mono">{REPO}</span>.</p>
  <div class="footer"><span>Notebook: Luan_Cesar_de_Souza_Nascimento_Atividade_4.ipynb</span><span>08</span></div>
</section>'''
    return head + cover + desafio + parte1 + montagem + parte2 + resultados + desafiadora + conclusao + '</body></html>'


def build():
    report = json.loads((RES / 'resultados.json').read_text(encoding='utf-8'))
    work = OUT / 'apresentacao_fonte'
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(ASSETS, work)
    figure(work / 'grafico_seed42.png')
    html = work / 'apresentacao.html'
    html.write_text(slides(report), encoding='utf-8')
    browser = next(p for p in BROWSERS if Path(p).exists())
    subprocess.run([browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    '--run-all-compositor-stages-before-draw', '--virtual-time-budget=5000',
                    f'--print-to-pdf={PDF}', html.as_uri()], check=True, capture_output=True, timeout=120)
    print('PDF gerado:', PDF)


if __name__ == '__main__':
    build()
