"""Posterior diagnostics, not a claim to measure occupied latent-space volume."""
import csv
import json
from pathlib import Path

import numpy as np


def latent_statistics(mu,logvar,active_threshold=.01):
    mu=np.asarray(mu,dtype=np.float64);logvar=np.asarray(logvar,dtype=np.float64)
    if mu.ndim!=2 or mu.shape!=logvar.shape or len(mu)<2 or mu.shape[1]<1:
        raise ValueError('Need matching [samples, latent_dims] arrays with at least two samples')
    if not np.isfinite(active_threshold) or active_threshold<0 or not np.isfinite(mu).all() or not np.isfinite(logvar).all():
        raise ValueError('Latents must be finite; active threshold must be nonnegative')
    with np.errstate(over='raise'):
        posterior_variance=np.exp(logvar)
    mean=mu.mean(0);variance=mu.var(0,ddof=1)
    kl=.5*(mu**2+posterior_variance-1-logvar).mean(0)
    covariance=np.atleast_2d(np.cov(mu,rowvar=False))
    eigenvalues=np.maximum(np.linalg.eigvalsh(covariance)[::-1],0)
    total=eigenvalues.sum()
    probabilities=eigenvalues/total if total>0 else np.zeros_like(eigenvalues)
    cumulative=np.cumsum(probabilities)
    participation=float(total**2/np.square(eigenvalues).sum()) if total>0 else 0.
    positive=probabilities[probabilities>0]
    entropy_rank=float(np.exp(-(positive*np.log(positive)).sum())) if total>0 else 0.
    summary=dict(samples=len(mu),latent_dims=mu.shape[1],active_threshold=active_threshold,
                 active_units=int((variance>active_threshold).sum()),
                 active_units_fraction=float((variance>active_threshold).mean()),
                 total_mean_kl_nats=float(kl.sum()),mean_kl_per_dimension_nats=float(kl.mean()),
                 effective_rank_participation=participation,effective_rank_entropy=entropy_rank,
                 principal_components_for_95_percent_variance=int(np.searchsorted(cumulative,.95)+1) if total>0 else 0,
                 max_observable_covariance_rank=min(len(mu)-1,mu.shape[1]),
                 interpretation='Active fraction is a threshold-dependent diagnostic, not fraction of latent volume occupied. High KL or active count alone does not prove useful capacity or justify a wider latent space.')
    dimensions=[dict(dimension=i,mu_mean=float(mean[i]),mu_variance=float(variance[i]),
                     mean_kl_nats=float(kl[i]),mean_posterior_variance=float(posterior_variance[:,i].mean()),
                     signal_to_posterior_noise=float(variance[i]/max(posterior_variance[:,i].mean(),1e-12)),
                     active=bool(variance[i]>active_threshold)) for i in range(mu.shape[1])]
    return summary,dimensions,eigenvalues


def save_latent_audit(mu,logvar,output_dir,metadata=None,active_threshold=.01,save_codes=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    summary,dimensions,eigenvalues=latent_statistics(mu,logvar,active_threshold)
    summary['metadata']=metadata or {}
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    with (output_dir/'dimensions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dimensions[0]));writer.writeheader();writer.writerows(dimensions)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].bar(range(len(dimensions)),[d['mu_variance'] for d in dimensions])
    axes[0,0].axhline(active_threshold,color='red',linestyle='--',label=f'Active threshold {active_threshold:g}')
    axes[0,0].set(title='Posterior mean variance by dimension',xlabel='Dimension',ylabel='Var(mu)');axes[0,0].legend()
    axes[0,1].bar(range(len(dimensions)),[d['mean_kl_nats'] for d in dimensions])
    axes[0,1].set(title='KL to N(0,1) by dimension',xlabel='Dimension',ylabel='Mean KL (nats)')
    cumulative=np.cumsum(eigenvalues)/max(eigenvalues.sum(),1e-12)
    axes[1,0].plot(range(1,len(eigenvalues)+1),cumulative);axes[1,0].axhline(.95,color='red',linestyle='--')
    axes[1,0].set(title='Covariance of posterior means',xlabel='Principal components',ylabel='Cumulative explained variance',ylim=(0,1.02))
    centered=np.asarray(mu,dtype=np.float64)-np.mean(mu,axis=0,dtype=np.float64)
    _,_,vectors=np.linalg.svd(centered,full_matrices=False)
    projected=np.einsum('ij,kj->ik',centered,vectors[:2],optimize=False)
    axes[1,1].scatter(projected[:,0],projected[:,1] if projected.shape[1]>1 else np.zeros(len(mu)),s=8,alpha=.4)
    axes[1,1].set(title='Posterior means: PCA projection (not full-space coverage)',xlabel='PC1',ylabel='PC2')
    fig.suptitle(f"Latent audit: {summary['active_units']}/{summary['latent_dims']} active; effective rank {summary['effective_rank_participation']:.2f}; N={summary['samples']}")
    fig.tight_layout();fig.savefig(output_dir/'latent_utilization.png',dpi=140);plt.close(fig)
    if save_codes:np.savez_compressed(output_dir/'latents.npz',mu=mu,logvar=logvar)
    (output_dir/'README.md').write_text(
        '# Latent utilization audit\n\n'+json.dumps(summary,indent=2)+'\n\n'
        'Use diverse held-out normal recordings for capacity decisions. Patches from one frame are correlated. '
        'Low effective rank with many active units indicates correlated directions, not necessarily a fully used bottleneck. '
        'Changing latent width requires a new model; compare reconstruction and labeled detection, not active fraction alone.\n')
    return summary
