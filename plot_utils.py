import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sb
from matplotlib.pyplot import rc_context
from scipy.stats import wilcoxon, norm
import scipy.stats as stats
import scanpy as sc
import anndata
import mudata
import math

mpl.rcParams['pdf.fonttype'] = 42


# ---------------------------------------------------------------------------
# Sample pairing
# ---------------------------------------------------------------------------

def getPairs():
    pairs = {
        'LS_1_PRE': 'LS_1_POST',
        'LS_2_PRE': 'LS_2_POST',
        'LS_3_PRE': 'LS_3_POST',
        'LS_4_PRE': 'LS_4_POST',
        'LS_5_PRE': 'LS_5_POST',
        'LS_6_PRE': 'LS_6_POST',
        'LS_7_PRE': 'LS_7_POST',
        'LS_8_PRE': 'LS_8_POST',
        'LS_9_PRE': 'LS_9_POST',
    }
    return pairs


# ---------------------------------------------------------------------------
# Boxplot (paired, with swarm + connecting lines)
# ---------------------------------------------------------------------------

def plotBoxplot(g1, g2, function=wilcoxon, title=None, ylabel=None, ptext='p = ',
                figsize=(3, 5), xticks=['Pre', 'Post'], palette=["#CC6699", "#66CCFF"],
                existingaxes=None):
    p_val = function(g1, g2)[1]

    if existingaxes is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = existingaxes
        plt.sca(existingaxes)

    g_box = sb.boxplot(
        (g1, g2),
        palette=palette,
        width=0.5,
        showcaps=True,
        boxprops={'linewidth': 1, 'zorder': 1, 'edgecolor': 'black'},
        whiskerprops={'linewidth': 1, 'color': 'black'},
        capprops={'linewidth': 1, 'color': 'black'},
        flierprops={'marker': '', 'alpha': 0},
        medianprops={'color': 'black', 'linewidth': 1},
        ax=existingaxes
    )
    g_box.grid(False)

    swarmax = sb.swarmplot((g1, g2), color='#000000', size=3, linewidth=0.3,
                           ax=existingaxes, edgecolor='black')

    pair1 = swarmax.collections[0].get_offsets()
    pair2 = swarmax.collections[1].get_offsets()

    for cur in range(len(pair1)):
        plt.plot(
            (pair1[cur][0], pair2[cur][0]),
            (pair1[cur][1], pair2[cur][1]),
            color='#4f4e4e',
            linewidth=0.5,
            alpha=0.6
        )

    miny = min(min(g1), min(g2))
    maxy = max(max(g1), max(g2))
    yrange = maxy - miny if maxy != miny else abs(maxy) if maxy != 0 else 1.0
    bracket_y = maxy + 0.05 * yrange
    text_y = maxy + 0.05 * yrange

    lbl = f"{ptext} {p_val:.3g}" if not np.isnan(p_val) else "n/a"
    plt.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + 0.02 * (bracket_y - maxy),
         bracket_y + 0.02 * (bracket_y - maxy), bracket_y],
        lw=1, c='k'
    )
    plt.text(0.5, text_y, lbl, ha='center', va='bottom', fontsize=10, color='k')

    plt.title(title)
    plt.ylabel(ylabel)

    sb.despine(ax=g_box, left=False, bottom=False)
    plt.xticks([0, 1], xticks, rotation=45)
    plt.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# Cell-type proportion boxplot
# ---------------------------------------------------------------------------

def plotProportionBoxplot(obj, annotationvar, annotation, ylabel='% PBMCs',
                          samplevar='SampleID', figsize=(3, 3), existingaxes=None):
    pairs = getPairs()

    g1, g2 = [], []
    for curkey in pairs:
        pre = 100 * sum(
            (obj.obs[annotationvar] == annotation) & (obj.obs[samplevar] == curkey)
        ) / sum(obj.obs[samplevar] == curkey)
        post = 100 * sum(
            (obj.obs[annotationvar] == annotation) & (obj.obs[samplevar] == pairs[curkey])
        ) / sum(obj.obs[samplevar] == pairs[curkey])
        g1.append(pre)
        g2.append(post)

    return plotBoxplot(g1, g2, title=annotation, ylabel=ylabel,
                       xticks=['Pre', 'Post'], figsize=figsize, existingaxes=existingaxes)


# ---------------------------------------------------------------------------
# Mean gene expression per sample
# ---------------------------------------------------------------------------

def getMeanPerSample(adata, genes):
    expr = adata[:, adata.var.index.isin(genes)].X.mean(1)
    if hasattr(expr, "toarray"):
        expr = expr.toarray().ravel()
    else:
        expr = np.asarray(expr).ravel()
    samples = adata.obs['SampleID'].values
    df_tmp = pd.DataFrame({'Names_groups': samples, 'Value': expr})
    return df_tmp.groupby('Names_groups', observed=False)['Value'].mean().reset_index()


# ---------------------------------------------------------------------------
# Gene expression boxplot (single gene or gene set)
# ---------------------------------------------------------------------------

def plotExpression(adata, pairs, genes, genename=None, title=None,
                   figsize=(2.2, 3.15), existingaxes=None):
    expression = getMeanPerSample(adata, genes)
    g1, g2 = [], []
    for curkey in pairs:
        g1.append(expression.loc[expression['Names_groups'] == curkey, 'Value'].values[0])
        g2.append(expression.loc[expression['Names_groups'] == pairs[curkey], 'Value'].values[0])

    rv = plotBoxplot(g1, g2, title=title, ylabel='Avg. Expression',
                     function=stats.wilcoxon, ptext="Wilcoxon P = ",
                     figsize=figsize, existingaxes=existingaxes)
    plt.tight_layout()
    return rv


# ---------------------------------------------------------------------------
# Multi-panel expression boxplots
# ---------------------------------------------------------------------------

def plotExpressionMultiple(obj, genes, ncols=2, figsize=(3, 3)):
    rnapairs = getPairs()
    nrows = int(np.ceil(len(genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for i, curgene in enumerate(genes):
        curi, curj = divmod(i, ncols)
        plotExpression(
            adata=obj,
            pairs=rnapairs,
            genes=[curgene],
            title=curgene,
            figsize=figsize,
            existingaxes=axes[curi, curj]
        )

    for i in range(len(genes), nrows * ncols):
        curi, curj = divmod(i, ncols)
        axes[curi, curj].axis("off")

    return fig


# ---------------------------------------------------------------------------
# Paired correlation scatter plot
# ---------------------------------------------------------------------------

def plotCorrelation(corrtable, g1name, g2name, hue, title, palette, ylim=None):
    fig = plt.figure(figsize=(5, 5))

    ax = sb.regplot(corrtable, x=g1name, y=g2name, scatter=None, color='#000000')
    sb.scatterplot(corrtable, x=g1name, y=g2name, hue=hue, palette=palette)
    print(title)
    print(stats.spearmanr(corrtable[g1name], corrtable[g2name]))
    print(stats.pearsonr(corrtable[g1name], corrtable[g2name]))

    ax.grid(False)

    pairs = getPairs()
    for curpair in pairs:
        p1 = corrtable.loc[curpair, [g1name, g2name]].values.ravel()
        p2 = corrtable.loc[pairs[curpair], [g1name, g2name]].values.ravel()
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], c='#000000', lw=0.5)

    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    return fig


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------

def _ensure_gene_index(df):
    if df.index.dtype.kind == 'O':
        return df.copy()
    for colname in ["Unnamed: 0", "gene", "Gene"]:
        if colname in df.columns:
            return df.copy().set_index(colname)
    raise ValueError("No gene name column found in DataFrame.")


def plot_volcano(degenes, title="", logfc=0.25, adjustp=0.05, label_fc=0.5, label_p=0.05,
                 selected_genes=None, color_values=('#cc6699', '#66ccff'),
                 figsize=(5, 5), x_lim=None, save_path=None, save_dpi=300):
    df = _ensure_gene_index(degenes)

    if 'log_p.adjust' not in df.columns:
        df = df.copy()
        df['log_p.adjust'] = -np.log10(df['FDR'])

    x = df['logFC']
    y = df['log_p.adjust']

    colors = np.full(len(df), '#CCCCCC', dtype=object)
    sig_mask = df['FDR'] < adjustp
    colors[np.where((x < -logfc) & sig_mask)] = color_values[0]
    colors[np.where((x >  logfc) & sig_mask)] = color_values[1]

    with rc_context({'figure.figsize': figsize, 'axes.grid': False, 'xtick.labelsize': 10}):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, c=colors, s=10, edgecolors='none')

        ax.axhline(-np.log10(adjustp), linestyle='--', lw=0.5, c='#CCCCCC')
        ax.axvline(-logfc, linestyle='--', lw=0.5, c='#CCCCCC')
        ax.axvline(+logfc, linestyle='--', lw=0.5, c='#CCCCCC')

        if x_lim is None:
            x_lim = np.nanmax(np.abs(x)) * 1.1
        ax.set_xlim(-x_lim, x_lim)

        if selected_genes is not None:
            labels_df = df.loc[df.index.intersection(selected_genes)]
        else:
            labels_df = df.loc[(np.abs(df['logFC']) >= label_fc) & (df['FDR'] < label_p)]

        for gene_name, row in labels_df.iterrows():
            ax.annotate(gene_name,
                        xy=(row['logFC'], row['log_p.adjust']),
                        xytext=(row['logFC'], row['log_p.adjust']),
                        fontsize=5)

        ax.set_xlabel('log2 fold-change')
        ax.set_ylabel('-log10(p-value)')
        ax.set_title(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=save_dpi, bbox_inches='tight')

        return fig


# ---------------------------------------------------------------------------
# Weighted KDE density on embedding
# ---------------------------------------------------------------------------

def add_genes_density_to_obs_wkde(adata, genes, embedding_key="X_umap", dims=(0, 1),
                                   n_grid=100, adjust=1.0, log_transform=False):
    if isinstance(genes, str):
        genes = [genes]

    coords = adata.obsm[embedding_key][:, list(dims)]
    x, y = coords[:, 0], coords[:, 1]
    n_cells = len(x)

    gx = np.linspace(x.min(), x.max(), n_grid)
    gy = np.linspace(y.min(), y.max(), n_grid)

    h_x = 1.06 * np.std(x) * (n_cells ** (-1 / 5)) * adjust
    h_y = 1.06 * np.std(y) * (n_cells ** (-1 / 5)) * adjust

    ax_grid = (gx[:, None] - x[None, :]) / h_x
    ay_grid = (gy[:, None] - y[None, :]) / h_y

    dnorm_ax = norm.pdf(ax_grid)
    dnorm_ay = norm.pdf(ay_grid)

    for gene in genes:
        expr = adata[:, gene].X
        expr = expr.toarray().ravel() if hasattr(expr, "toarray") else np.asarray(expr).ravel()

        if log_transform:
            expr = np.log1p(expr)

        w = expr.copy()
        sum_w = w.sum()

        if sum_w == 0:
            cell_density = np.zeros(n_cells)
        else:
            A = dnorm_ax * w[None, :]
            B = dnorm_ay * w[None, :]
            z = np.dot(A, B.T) / (sum_w * h_x * h_y)

            ix = np.clip(np.searchsorted(gx, x, side="right") - 1, 0, n_grid - 1)
            iy = np.clip(np.searchsorted(gy, y, side="right") - 1, 0, n_grid - 1)
            cell_density = z[ix, iy]

        adata.obs[f"density_{gene}"] = cell_density


# ---------------------------------------------------------------------------
# Pathway score
# ---------------------------------------------------------------------------

def compute_pathway_score(adata, gene_list, pathway_name, min_detected_percent=5):
    min_cells = int((min_detected_percent / 100) * adata.n_obs)
    gene_detected = np.array((adata.X > 0).sum(axis=0)).flatten()

    if gene_detected.shape[0] != len(adata.var_names):
        gene_detected = gene_detected.T

    filtered_genes = [
        g for g in gene_list
        if g in adata.var_names and gene_detected[adata.var_names.get_loc(g)] >= min_cells
    ]

    if filtered_genes:
        adata.obs[f'{pathway_name} Pathway score'] = np.array(
            adata[:, filtered_genes].X.mean(axis=1)
        ).flatten()
        print(f"Computed {pathway_name} Pathway Score using {len(filtered_genes)} genes.")
        print(filtered_genes)
    else:
        print(f"No genes in {pathway_name} passed the detection threshold.")


# ---------------------------------------------------------------------------
# eRegulon object (SCENIC+)
# ---------------------------------------------------------------------------

def getEregulonObj(h5mufile):
    scplus_mdata = mudata.read(h5mufile)
    eRegulon_gene_AUC = anndata.concat(
        [scplus_mdata["direct_gene_based_AUC"], scplus_mdata["extended_gene_based_AUC"]],
        axis=1,
    )
    eRegulon_gene_AUC.obs = scplus_mdata.obs.loc[eRegulon_gene_AUC.obs_names]
    sc.pp.neighbors(eRegulon_gene_AUC, use_rep="X")
    sc.tl.umap(eRegulon_gene_AUC)

    eRegulon_gene_AUC.obs['Groups'] = 'Baseline'
    for cur in eRegulon_gene_AUC.obs.index:
        if '6M' in cur:
            eRegulon_gene_AUC.obs.loc[cur, 'Groups'] = '6M'
    return eRegulon_gene_AUC


# ---------------------------------------------------------------------------
# Regulon violin plot
# ---------------------------------------------------------------------------

def plotRegulonViolin(regulonobject, eregulon, axes, pregroup='Baseline', postgroup='6M',
                      palette=None, ptext='p = '):
    plt.sca(axes)

    pre = regulonobject.X[:, regulonobject.var.index.isin([eregulon])][
        (regulonobject.obs['Groups'] == pregroup)
    ]
    post = regulonobject.X[:, regulonobject.var.index.isin([eregulon])][
        (regulonobject.obs['Groups'] == postgroup)
    ]

    pre_df = pd.DataFrame(pre, columns=[eregulon])
    pre_df['Group'] = 'Pre'
    post_df = pd.DataFrame(post, columns=[eregulon])
    post_df['Group'] = 'Post'
    df = pd.concat((pre_df, post_df))

    p_val = stats.mannwhitneyu(pre, post)[1][0]

    axes.yaxis.grid(False)
    axes.xaxis.grid(False)
    sb.violinplot(df, x='Group', y=eregulon, hue='Group', palette=palette, ax=axes)

    miny = min(min(pre), min(post))
    maxy = max(max(pre), max(post))
    yrange = maxy - miny if maxy != miny else abs(maxy) if maxy != 0 else 1.0
    bracket_y = maxy + 0.05 * yrange
    text_y = maxy + 0.05 * yrange

    lbl = f"{ptext} {p_val:.3g}" if not np.isnan(p_val) else "n/a"
    plt.plot(
        [0, 0, 1, 1],
        [bracket_y, bracket_y + 0.02 * (bracket_y - maxy),
         bracket_y + 0.02 * (bracket_y - maxy), bracket_y],
        lw=1, c='k'
    )
    plt.text(0.5, text_y, lbl, ha='center', va='bottom', fontsize=10, color='k')
    plt.tight_layout()