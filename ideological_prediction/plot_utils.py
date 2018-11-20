import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from os import path
import pandas as pd
import pickle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from dimensional_structure.plot_utils import get_short_names, plot_loadings
from selfregulation.utils.plot_utils import beautify_legend, CurvedText, format_num, save_figure

colors = sns.color_palette('Blues_d',3) + sns.color_palette('Reds_d',2)[:1]
shortened_factors = get_short_names()

def visualize_importance(importance, ax, xticklabels=True, yticklabels=True, 
                         axes_linewidth=None, label_size=10,
                         label_scale=0, title=None, 
                         ymax=None, color=colors[1],
                         show_sign=True):
    importance_vars = importance[0]
    importance_vars = [shortened_factors.get(v,v) for v in importance_vars]
    if importance[1] is not None:
        importance_vals = [abs(i) for i in importance[1]]
        plot_loadings(ax, importance_vals, kind='line', offset=.5, 
                      colors=[color], plot_kws={'alpha': 1, 
                                                 'linewidth': label_size/4})
    else:
        ax.set_yticks([])
    ax.grid(linewidth=label_size/8)
    if axes_linewidth:
        plt.setp(list(ax.spines.values()), linewidth=axes_linewidth)

    # set up x ticks
    xtick_locs = np.arange(0.0, 2*np.pi, 2*np.pi/len(importance_vars))
    ax.set_xticks(xtick_locs)
    ax.set_xticks(xtick_locs+np.pi/len(importance_vars), minor=True)
    if xticklabels:
        if type(importance_vars[0]) != str:
            importance_vars = ['Fac %s' % str(i+1) for i in importance_vars]
        ticks = importance_vars
        scale = 1+label_scale
        size = ax.get_position().expanded(scale, scale)
        ax2=ax.get_figure().add_axes(size,zorder=2)
        for i, text in enumerate(ticks):
            fontcolor='k'
            if importance[1][i] < 0 and show_sign:
                fontcolor = 'r'
            arc_start = (i+.1)*2*np.pi/len(importance_vars)
            arc_end = (i+.9)*2*np.pi/len(importance_vars)
            curve = [
                .85*np.cos(np.linspace(arc_start,arc_end,100)),
                .85*np.sin(np.linspace(arc_start,arc_end,100))
            ]  
            plt.plot(*curve, alpha=0)
            curvetext = CurvedText(
                x = curve[0][::-1],
                y = curve[1][::-1],
                text=text, #'this this is a very, very long text',
                va = 'bottom',
                axes = ax2,
                fontsize=label_size,
                color=fontcolor
            )
            ax2.set_xlim([-1,1]); ax2.set_ylim([-1,1])
            ax2.axis('off')
        
    if title:
        ax.set_title(title, fontsize=label_size*1.5, y=1.1)
    # set up yticks
    if len(importance[1]) != 0:
        ax.set_ylim(bottom=0)
        if ymax:
            ax.set_ylim(top=ymax)
        ytick_locs = ax.yaxis.get_ticklocs()
        new_yticks = np.linspace(0, ytick_locs[-1], 7)
        ax.set_yticks(new_yticks)
        if yticklabels:
            labels = np.round(new_yticks,2)
            replace_dict = {i:'' for i in labels[::2]}
            labels = [replace_dict.get(i, i) for i in labels]
            ax.set_yticklabels(labels)
    # optional to shade to show sign of beta value
    if show_sign:
        data_coords = ax.lines[0].get_data()
        ylim = ax.get_ylim()
        gap = data_coords[0][1]-data_coords[0][0]
        centers = []
        for i, val in enumerate([j for j in importance[1]]):
            if val<0:
                centers.append(data_coords[0][i])
        for center in centers:
            ax.axvspan(xmin=center-gap/2, xmax=center+gap/2,
                       ymin=ylim[0], ymax=ylim[1]+1,
                       facecolor='r', alpha=.1)

def plot_prediction(predictions, shuffled_predictions, 
                    target_order=None, 
                    metric='R2', size=4.6,  
                    dpi=300, filename=None):
    """ Plots predictions resulting from "run_prediction" function
    
    Args:
        predictions: dictionary of run_prediction results
        shuffled_predictions: dictionary of run_prediction shuffled results
        target_order: (optional) a list of targets to order the plot
        metric: which metric from the output of run_prediction to use
        size: figure size
        dpi: dpi to use for saving
        ext: extension to use for saving (e.g., pdf)
        filename: if provided, save to this location
    """
    colors = sns.color_palette('Blues_d',4)
    basefont = max(size, 5)
    sns.set_style('white')
    if target_order is None:
        target_order = predictions.keys()
    prediction_keys = predictions.keys()
    # get prediction success
    # plot
    shuffled_grey = [.3,.3,.3]
    # plot variables
    figsize = (size, size*.75)
    fig = plt.figure(figsize=figsize)
    # plot bars
    width=.23
    ax1 = fig.add_axes([0,0,1,.5]) 
    for predictor_i, key in enumerate(prediction_keys):
        prediction = predictions[key]
        shuffled_prediction = shuffled_predictions[key]
        r2s = [[k,prediction[k]['scores_cv'][0][metric]] for k in target_order]
        # get shuffled values
        shuffled_r2s = []
        for i, k in enumerate(target_order):
            # normalize r2s to significance
            R2s = [i[metric] for i in shuffled_prediction[k]['scores_cv']]
            R2_95 = np.percentile(R2s, 95)
            shuffled_r2s.append((k,R2_95))
        # convert nans to 0
        r2s = [(i, k) if k==k else (i,0) for i, k in r2s]
        shuffled_r2s = [(i, k) if k==k else (i,0) for i, k in shuffled_r2s]
        
        ind = np.arange(len(r2s))
        ax1.bar(ind+width*predictor_i, [i[1] for i in r2s], width, 
                label='%s Prediction' % ' '.join(key.title().split('_')), 
                color=colors[predictor_i])
        # plot shuffled values above
        if predictor_i == len(prediction_keys)-1:
            shuffled_label = '95% shuffled prediction'
        else:
            shuffled_label = None
        ax1.bar(ind+width*predictor_i, [i[1] for i in shuffled_r2s], width, 
                 color='none', edgecolor=shuffled_grey, 
                linewidth=size/10, linestyle='--', label=shuffled_label)
        
    ax1.set_xticks(np.arange(0,len(r2s))+width/2)
    ax1.set_xticklabels(['\n'.join(i[0].split()) for i in r2s], 
                        rotation=0, fontsize=basefont*.75, ha='center')
    ax1.tick_params(axis='y', labelsize=size*1.2)
    ax1.tick_params(length=size/2, width=size/10, pad=size/2, bottom=True, left=True)
    xlow, xhigh = ax1.get_xlim()
    if metric == 'R2':
        ax1.set_ylabel(r'$R^2$', fontsize=basefont*1.5, labelpad=size*1.5)
    else:
        ax1.set_ylabel(metric, fontsize=basefont*1.5, labelpad=size*1.5)
    # add a legend
    leg = ax1.legend(fontsize=basefont*1.4, loc='upper right', 
                     bbox_to_anchor=(1.1, 1.1), frameon=True, 
                     handlelength=0, handletextpad=0, framealpha=1)
    beautify_legend(leg, colors[:len(predictions)]+[shuffled_grey])
    # draw grid
    ax1.set_axisbelow(True)
    plt.grid(axis='y', linestyle='dotted', linewidth=size/6)
    plt.setp(list(ax1.spines.values()), linewidth=size/10)
    if filename is not None:
        save_figure(fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    else:
        return fig

def plot_prediction_scatter(predictions, predictors, targets, 
                            target_order=None, metric='R2', size=4.6,  
                            dpi=300, filename=None):
    # subset predictors
    predictors = predictors.loc[targets.index]
    if target_order is None:
        target_order = predictions.keys()
        
    sns.set_style('white')
    n_cols = 4
    n_rows = math.ceil(len(target_order)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows))
    axes = fig.get_axes()
    for i,v in enumerate(target_order):
        MAE = format_num(predictions[v]['scores_cv'][0]['MAE'])
        R2 = format_num(predictions[v]['scores_cv'][0]['R2'])
        axes[i].set_title('%s\nR2: %s, MAE: %s' % (v, R2, MAE), 
            fontweight='bold', fontsize=size*1.2)
        clf=predictions[v]['clf']
        axes[i].scatter(targets[v], clf.predict(predictors), s=size*2.5,
                        edgecolor='white', linewidth=size/30)  
        axes[i].tick_params(length=0, labelsize=0)
        # add diagonal
        xlim = axes[i].get_xlim()
        ylim = axes[i].get_ylim()
        axes[i].plot(xlim, ylim, ls="-", c=".5", zorder=-1)
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim)
        for spine in ['top', 'right']:
            axes[i].spines[spine].set_visible(False)
        if i%n_cols==0:
            axes[i].set_ylabel('Predicted Score', fontsize=size*1.2)
    for ax in axes[-(len(target_order)+1):]:
        ax.set_xlabel('Target Score', fontsize=size*1.2)
    
    empty_plots = n_cols*n_rows - len(targets.columns)
    for ax in axes[-empty_plots:]:
        ax.set_visible(False)
    plt.subplots_adjust(hspace=.4, wspace=.3)
    
    if filename is not None:
        save_figure(fig, filename, {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
        
def plot_prediction_comparison(results, size=4.6, change=False,
                               dpi=300, ext='png', plot_dir=None):
    R2s = {}
    for EFA in [False, True]:
        predictions = results.get_prediction_files(EFA=EFA, change=change, 
                                                   shuffle=False)
        predictions = sorted(predictions, key = path.getmtime)
        classifiers = np.unique([i.split('_')[-2] for i in predictions])
        # get last prediction file of each type
        for classifier in classifiers:
            filey = [i for i in predictions if classifier in i][-1]
            prediction_object = pickle.load(open(filey, 'rb'))['data']
            R2 = [i['scores_cv'][0]['R2'] for i in prediction_object.values()]
            R2 = np.nan_to_num(R2)
            feature = 'EFA' if EFA else 'IDM'
            R2s[feature+'_'+classifier] = R2

    R2s = pd.DataFrame(R2s).melt(var_name='Classifier', value_name='R2')
    R2s['Feature'], R2s['Classifier'] = R2s.Classifier.str.split('_', 1).str
    f = plt.figure(figsize=(size, size*.62))
    sns.barplot(x='Classifier', y='R2', data=R2s, hue='Feature',
                palette=colors[1:3], errwidth=size/5)
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=size*1.8)
    ax.tick_params(axis='x', labelsize=size*1.8)
    leg = ax.legend(fontsize=size*2, loc='upper right')
    beautify_legend(leg, colors[1:3])
    plt.xlabel('Classifier', fontsize=size*2.2, labelpad=size/2)
    plt.ylabel('R2', fontsize=size*2.2, labelpad=size/2)
    plt.title('Comparison of Prediction Methods', fontsize=size*2.5, y=1.05)
    
    if plot_dir is not None:
        filename = 'prediction_comparison.%s' % ext
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    
def plot_prediction_relevance(results, EFA=True, classifier='ridge', 
                              rotate='oblimin', change=False, size=4.6, 
                              dpi=300, ext='png', plot_dir=None):
    """ Plots the relevant relevance of each factor for predicting all outcomes """
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)['data']

    targets = list(predictions.keys())
    predictors = predictions[targets[0]]['predvars']
    importances = abs(np.vstack([predictions[k]['importances'] for k in targets]))
    # scale to 0-1 
    scaler = MinMaxScaler()
    scaled_importances = scaler.fit_transform(importances.T).T
    # make proportion
    scaled_importances = scaled_importances/np.expand_dims(scaled_importances.sum(1),1)
    # convert to dataframe
    scaled_df = pd.DataFrame(scaled_importances, index=targets, columns=predictors)
    melted = scaled_df.melt(var_name='Factor', value_name='Importance')
    plt.figure(figsize=(8,12))
    f=sns.boxplot(y='Factor', x='Importance',  data=melted,
                  width=.5)
    if plot_dir is not None:
        filename = 'prediction_relevance'
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()

def plot_outcome_ontological_similarity(results, EFA=True, classifier='ridge', 
                                        rotate='oblimin', change=False, size=4.6, 
                                        dpi=300, ext='png',  plot_dir=None):
    """ plots similarity of ontological fingerprints between outcomes """
    predictions = results.load_prediction_object(EFA=EFA, 
                                                 change=change,
                                                 classifier=classifier,
                                                 rotate=rotate)['data']

    targets = list(predictions.keys())
    predictors = predictions[targets[0]]['predvars']
    importances = np.vstack([predictions[k]['importances'] for k in targets])
    # convert to dataframe
    df = pd.DataFrame(importances, index=targets, columns=predictors)
    plt.figure(figsize=(8,12))
    f=sns.clustermap(df.T.corr(),
                     cmap=sns.diverging_palette(220,15,n=100,as_cmap=True))
    ax = f.ax_heatmap
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if plot_dir is not None:
        filename = 'prediction_relevance'
        save_figure(f, path.join(plot_dir, filename), 
                    {'bbox_inches': 'tight', 'dpi': dpi})
        plt.close()
    