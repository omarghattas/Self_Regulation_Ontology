from fancyimpute import SoftImpute
from itertools import product
from os import path
import pandas as pd

from dimensional_structure.utils import hierarchical_cluster
from dimensional_structure.prediction_utils import run_prediction
from dimensional_structure.prediction_plots import plot_prediction
from selfregulation.utils.plot_utils import  save_figure
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_recent_dataset, get_info

dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv')
results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
plot_dir = path.join(results_dir, 'Plots')

# run prediction
target_name = 'ideology'
target = ideo_data+1E-10 # adding a small float causes values to be seen as continuous by "type_of_target"
shuffle_reps = 2

# define predictors
survey_scores = results['survey'].EFA.get_scores()
task_scores = results['task'].EFA.get_scores()
predictors = {'survey': survey_scores,
              'task': task_scores,
              'demographics': results['task'].DA.get_scores(),
              'full_ontology': pd.concat([survey_scores, task_scores], axis=1)}
# define targets
targets = {'ideo_factors': ideo_data.filter(regex='Factor'),
           'ideo_orientations': ideo_data.drop(ideo_data.filter(regex='Factor|SECS').columns, axis=1),
           'ideo_policies': ideo_data.filter(regex='SECS')}
for key, target in targets.items():
    imputed = pd.DataFrame(SoftImpute().complete(target),
                            index=target.index,
                            columns=target.columns)
    targets[key] = imputed

# do prediction with ontological factors
predictions = {}
shuffled_predictions = {}
classifier = 'ridge'
for predictor_key, scores in predictors.items():
    for target_key, target in targets.items():
        predictions[(predictor_key, target_key)] = \
                    run_prediction(scores, 
                                   target, 
                                   results_dir,
                                   outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                   shuffle=False,
                                   classifier=classifier, 
                                   verbose=True,
                                   save=True,
                                   binarize=False)['data']
        shuffled_predictions[(predictor_key, target_key)] = \
                            run_prediction(scores, 
                                           target, 
                                           results_dir,
                                           outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                           shuffle=shuffle_reps,
                                           classifier=classifier, 
                                           verbose=True,
                                           save=True,
                                           binarize=False)['data']
        
# predictions with raw measures
predictor_key = 'raw_measures'
scores = get_behav_data(file='meaningful_variables_imputed.csv')
classifier = 'lasso'
for target_key, target in targets.items():
    predictions[(predictor_key, target_key)] = \
                run_prediction(scores, 
                               target, 
                               results_dir,
                               outfile='%s_%s_prediction' % (predictor_key, target_key), 
                               shuffle=False,
                               classifier=classifier, 
                               verbose=True,
                               save=True,
                               binarize=False)['data']
    shuffled_predictions[(predictor_key, target_key)] = \
                        run_prediction(scores, 
                                       target, 
                                       results_dir,
                                       outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                       shuffle=shuffle_reps,
                                       classifier=classifier, 
                                       verbose=True,
                                       save=True,
                                       binarize=False)['data']                           

for target in targets.keys():
    imputed = targets[target]
    target_order = None
    if imputed.shape[1] > 3:
        clustering = hierarchical_cluster(imputed.T, method='average',
                                      pdist_kws={'metric': 'abscorrelation'})
        clustered_df = clustering['clustered_df']
        target_order = clustered_df.columns
    for predictor in predictors.keys():
        key = (predictor, target)
        key_name = '%s_%s' % key
        if predictor == 'demographics':
            EFA = results['task'].DA
        elif predictor == "full_ontology":
            EFA = True
        else:
            EFA = results[predictor].EFA
        fig = plot_prediction(predictions[key], shuffled_predictions[key], 
                              EFA=EFA, target_order=target_order, 
                              show_sign=True, size=15)
        # after base plot modifications
        ylim = fig.axes[0].get_ylim()
        fig.axes[0].set_ylim(ylim[0], .45)
        save_figure(fig, '/home/ian/tmp/%s_prediction' % key_name, 
                    {'bbox_inches': 'tight', 'dpi': 300})
        