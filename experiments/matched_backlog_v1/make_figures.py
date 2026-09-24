"""Export matched-budget heldout comparisons and the complete selection grid."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];RESULTS=ROOT/'experiments/results/matched_backlog_v1'
OUT=RESULTS/'analysis/figures'
METHODS=['QAPG-R','QAPG-capped','NoQuad-capped','GUPA-O-capped','EEDO-adapted','BP-Greedy-capped']
LABELS=['QAPG-R','Original QAPG (capped)','NoQuad (capped)','GUPA-O (capped)','EEDO adaptation','BP-Greedy (capped)']
COLORS=['#2368A2','#DD7834','#B89828','#7A8441','#AD668D','#565B63']
MARKERS=['o','s','^','D','v','P'];STYLES=['-','--','-.',':','--','-.']
def read(p):
    with p.open() as h:return list(csv.DictReader(h))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((RESULTS/'analysis/ANALYSIS.json').read_text())['status']=='ANALYZED'
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#6C7075','axes.labelcolor':'#292D32','text.color':'#292D32','xtick.color':'#42474D','ytick.color':'#42474D','pdf.fonttype':42,'svg.fonttype':'none'})
    rows=read(RESULTS/'analysis/HOLDOUT_COMPARISON.csv')
    handles=[Line2D([0],[0],color=c,marker=m,linestyle=s,label=l,markersize=5) for c,m,s,l in zip(COLORS,MARKERS,STYLES,LABELS)]
    figures=[]
    for kind in ['heldout','selection']:
        fig,axes=plt.subplots(1,3,figsize=(10.8,4.3));fig.subplots_adjust(left=.09,right=.99,bottom=.25,top=.74,wspace=.30)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,.925),ncol=3,frameon=False,columnspacing=1.9,handlelength=2.4)
        for beta,ax in enumerate(axes,1):
            ax.set_title(r'$\beta='+str(beta)+r'$');ax.grid(axis='y',color='#E7E8EB',linewidth=.7,zorder=0)
            if kind=='heldout':
                for method,c,m,s in zip(METHODS,COLORS,MARKERS,STYLES):
                    subset=sorted([r for r in rows if r['method']==method and int(r['beta'])==beta],key=lambda r:float(r['rho']))
                    x=np.array([float(r['rho']) for r in subset]);y=np.array([float(r['energy_mean']) if r['status']=='PASS' else np.nan for r in subset]);e=np.array([float(r['energy_sd']) if r['status']=='PASS' else np.nan for r in subset])
                    ax.errorbar(x,y,yerr=e,color=c,marker=m,linestyle=s,markersize=5,linewidth=1.3,capsize=2,zorder=3)
                    failed=[r for r in subset if r['status']=='TEST_BACKLOG_FAIL']
                    if failed:ax.scatter([float(r['rho']) for r in failed],[float(r['energy_mean']) for r in failed],marker='x',s=65,color=c,linewidth=1.7,zorder=5)
                ax.set_xlim(1.3,5.2);ax.set_xticks([1.5,2,3,5]);ax.set_ylim(bottom=0)
                ax.set_xlabel(r'Backlog budget $\rho$ (arrival slots)')
            else:
                grid=read(RESULTS/'selection/selection_grid.csv')
                for method,c,m,s in zip(METHODS,COLORS,MARKERS,STYLES):
                    subset=sorted([r for r in grid if r['method']==method and int(r['beta'])==beta],key=lambda r:int(r['scale_index']))
                    ax.plot([float(r['mean_backlog_mbit']) for r in subset],[float(r['mean_energy_j_per_slot']) for r in subset],color=c,marker=m,linestyle=s,markersize=4,linewidth=1.1,zorder=3)
                ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Mean aggregate backlog (Mbit)')
                ax.grid(which='minor',axis='y',color='#F2F3F5',linewidth=.5)
            if beta==1:ax.set_ylabel('Mean energy (J/slot)')
        if kind=='heldout':
            fig.suptitle('Energy under matched aggregate-backlog budgets',y=.985,fontsize=13)
            fail=sum(r['status']=='TEST_BACKLOG_FAIL' for r in rows);nf=sum(r['status']=='NO_FEASIBLE_GRID_POINT' for r in rows)
            note=f'Mean ± sample SD over 5 held-out seeds; 3000 slots, first 300 excluded. Budget = 4.8βρ Mbit.\nTest failures: {fail}. Original QAPG: no grid-feasible point at β=3, ρ=1.5, 2, 3. Lines are guides.'
            stem='heldout_energy_at_matched_backlog'
        else:
            fig.suptitle('Energy–backlog trade-offs across the selection grid',y=.985,fontsize=13)
            note='Selection data: 2 seeds × 9 energy weights per method and load; 1500 slots, first 300 excluded.\nAll 162 grid settings shown. Both axes are logarithmic. Curves connect increasing energy weights; not a held-out frontier.'
            stem='selection_energy_backlog_grid'
        fig.text(.07,.075,note,ha='left',va='center',fontsize=8.3,color='#4F5660',linespacing=1.6)
        for ext in ['pdf','svg','png']:
            path=OUT/f'{stem}.{ext}';fig.savefig(path,dpi=180,facecolor='white');figures.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        plt.close(fig)
    (OUT/'FIGURE_MANIFEST.json').write_text(json.dumps({'matplotlib':matplotlib.__version__,'numpy':np.__version__,'plotter_sha256':sha(Path(__file__)),'inputs':{str(p.relative_to(ROOT)):sha(p) for p in [RESULTS/'analysis/HOLDOUT_COMPARISON.csv',RESULTS/'selection/selection_grid.csv']},'figures':figures},indent=2))
    print(json.dumps(figures,indent=2))
if __name__=='__main__':main()
