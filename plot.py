'''
Print the results in the paper and plot Figure 3 from the output of main.py
'''
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pylab as plt

seeds = [1,2,7]

#Load the results of one experiment for each seed (named as in main.model_run)
def load(variant='flow',sigma=0,layers=None,size1=2**3,operator='fine'):
    results = []
    for seed in seeds:
        name = 'results/%s_sigma%g_seed%d' % (variant,sigma,seed)
        if layers is not None:
            name += '_L%d' % layers
        if size1!=2**3:
            name += '_size%d-%d' % (size1,2**6)
        if operator!='fine':
            name += '_' + operator
        results.append(np.load(name+'.npz'))
    return results

#Squared L2 error for each seed (mean over runs and time levels)
def error(results):
    return np.array([r['L2err'].mean() for r in results])

#Mean and (population) standard deviation over seeds, as in the paper
def fmt(e):
    exponent = int(('%.2e' % e.mean()).split('e')[1])
    return '(%.2f +- %.2f)e%03d' % (e.mean()/10**exponent, e.std()/10**exponent, exponent)

#Percentage of the validation runs, for each seed, on which a has a smaller error than b
def improved(a,b):
    return ', '.join('%.1f%%' % (100*np.mean(x['L2err'].mean(1) < y['L2err'].mean(1)))
                     for x, y in zip(a,b))


#Table 1, with the energy deviation and the coarse error quoted in the text
print('Table 1: square of the L2 error on the validation data, mean +- standard deviation' +
      ' over seeds', seeds)
for sigma in [0,0.1,1,10]:
    results = load(sigma=sigma)
    print('  sigma = %-4g %s,  relative deviation of the energy from the initial energy %.1e'
          % (sigma, fmt(error(results)), np.mean([r['dEnergy'].mean() for r in results])))
results = load()
coarse = np.mean([r['L2coarse'].mean() for r in results])
print('Coarse solution interpolated onto the fine mesh: square of the L2 error %.2e,' % coarse +
      ' %.1e times that with sigma = 0' % (coarse/error(results).mean()))
print('Relative L2 error: %.3f%% with sigma = 0, %.1f%% for the coarse solution' % (
    100*np.mean([np.sqrt(r['L2err'].sum()/r['L2norm'].sum()) for r in results]),
    100*np.mean([np.sqrt(r['L2coarse'].sum()/r['L2norm'].sum()) for r in results])))

#Figure 3 and the comparisons in the text, with sigma = 0
print('\nFigure 3: square of the L2 error, mean +- standard deviation over seeds')
print('  %-5s %-4s  %-21s %-21s %-21s %-12s %s' % ('s1', 'L', 'UNet alone', 'without flow map',
                                                    'with flow map', 'with/without',
                                                    'runs improved by the flow map'))
for size1, layers in [(2**3,2), (2**3,3), (2**3,4), (2**3,5), (2**3,None),
                      (2**5,2), (2**5,3)]:
    flow = load('flow',layers=layers,size1=size1)
    noflow = load('noflow',layers=layers,size1=size1)
    unet_error = '-' #not run
    if size1==2**3:
        unet_error = fmt(error(load('unet',layers=layers)))
    L = layers
    if layers is None:
        L = 'full'
    print('  %-5s %-4s  %-21s %-21s %-21s %-12.2f %s' % (
        '2^%dN' % np.log2(size1), L, unet_error,
        fmt(error(noflow)), fmt(error(flow)), error(flow).mean()/error(noflow).mean(),
        improved(flow,noflow)))
print('(2^5N is in Appendix B)')

print('\nAppendix B: parameters with the flow map, without it, and of UNet alone')
for layers in [2,3,4,5,None]:
    L = layers
    if layers is None:
        L = 'full'
    print('  L = %s: %d, %d, %d' % (L, load('flow',layers=layers)[0]['params'],
                                     load('noflow',layers=layers)[0]['params'],
                                     load('unet',layers=layers)[0]['params']))

print('\nAppendix B: UNet alone with about as many parameters as the network with the flow map')
for layers, size1 in [(2,2**4), (3,12)]:
    flow = load('flow',layers=layers)
    unet = load('unet',layers=layers,size1=size1)
    print('  L = %d: %d against %d parameters, %.1f times the error' % (
        layers, unet[0]['params'], flow[0]['params'], error(unet).mean()/error(flow).mean()))
    print('    runs improved by the flow map: %s' % improved(flow,unet))

print('\nAppendix B: the flow map replaced by the identity or a random map S Q S^-1, as a' +
      ' multiple of the error with the flow map (and without the flow map, for comparison)')
for layers in [2,3]:
    flow = load('flow',layers=layers)
    flow_identity = load('flow',layers=layers,operator='identity')
    flow_random = load('flow',layers=layers,operator='random')
    noflow = load('noflow',layers=layers)
    flow_error = error(flow).mean()
    print('  L = %d: identity %.1f, random %.1f, without flow map %.1f' % (
        layers, error(flow_identity).mean()/flow_error, error(flow_random).mean()/flow_error,
        error(noflow).mean()/flow_error))
    print('    runs improved by the flow map: %s (identity), %s (random)' % (
        improved(flow,flow_identity), improved(flow,flow_random)))

#Draw Figure 3 at its printed size (0.45 of the text width)
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 8,
                     'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
depths = [2,3,4,5]
styles = {'unet': ('tab:gray', 's', r'\texttt{UNet} alone'),
          'noflow': ('tab:orange', '^', 'without flow map'),
          'flow': ('tab:blue', 'o', 'with flow map')}
fig, (ax, axf) = plt.subplots(1, 2, figsize=(0.45*6.5, 1.6), sharey=True, layout='constrained',
                              gridspec_kw={'width_ratios': [4, 1]})
errors = {variant: [error(load(variant,layers=layers)) for layers in depths]
          for variant in styles}
mean = {variant: np.array([x.mean() for x in errors[variant]]) for variant in styles}
#Shade the gain from the flow map
ax.fill_between(depths, mean['flow'], mean['noflow'], color='tab:blue', alpha=0.12, linewidth=0)
for k, (variant, (color, marker, label)) in enumerate(styles.items()):
    ax.plot(depths, mean[variant], color=color, marker=marker, markersize=4, linewidth=1,
            label=label)
    ax.errorbar(depths, mean[variant], yerr=[x.std() for x in errors[variant]], color=color,
                linestyle='none')
    #The full UNet on the right, offset so that markers do not overlap
    full = error(load(variant))
    axf.errorbar((k-1)*0.3, full.mean(), yerr=full.std(), color=color, marker=marker,
                 markersize=4, linestyle='none')
axf.set_xlim(-0.7, 0.7)
axf.set_xticks([0])
axf.set_xticklabels(['full'])
axf.spines['left'].set_visible(False)
axf.tick_params(axis='y', which='both', left=False)
#Break marks on the x-axis between the panels
for a, x in [(ax, 1), (axf, 0)]:
    a.plot([x], [0], transform=a.transAxes, marker=[(-1, -1.5), (1, 1.5)], markersize=5,
           linestyle='none', color='k', mew=0.8, clip_on=False)
ax.set_yscale('log')
ax.set_ylim(2e-8, 1.5e-4)
ax.set_xticks(depths)
ax.set_xlim(1.8, 5.2)
ax.set_xlabel(r'Convolutions $L$ in each \texttt{UNet}')
ax.set_ylabel(r'Squared $L_2$ error')
#Legend outside the layout, so that the panels keep their widths
axf.legend(*ax.get_legend_handles_labels(), loc='upper right', frameon=False).set_in_layout(False)
filename = 'results/Figure3.pdf'
fig.savefig(filename)
print('\nFigure saved as %s' % filename)
