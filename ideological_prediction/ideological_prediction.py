# remove sklearn deprecation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from fancyimpute import SoftImpute
from os import makedirs, path
import pandas as pd
import pickle

from dimensional_structure.prediction_utils import run_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_recent_dataset, get_info

dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv')
results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
makedirs(results_dir, exist_ok=True)

# run prediction
target_name = 'ideology'
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
    targets[key] = imputed+1E-5

# do prediction with ontological factors
predictions = {}
shuffled_predictions = {}
classifier = 'ridge'
for predictor_key, scores in predictors.items():
    for target_key, target in targets.items():
        print('*'*79)
        print('Running Prediction: %s predicting %s' % (predictor_key, target_key))
        print('*'*79)
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
DV_scores = get_behav_data(file='meaningful_variables_imputed.csv')
predictors['raw_measures'] = DV_scores
classifier = 'lasso'
for target_key, target in targets.items():
    print('Running Prediction: raw measures predicting %s' % target_key)
    predictions[(predictor_key, target_key)] = \
                run_prediction(DV_scores, 
                               target, 
                               results_dir,
                               outfile='%s_%s_prediction' % (predictor_key, target_key), 
                               shuffle=False,
                               classifier=classifier, 
                               verbose=True,
                               save=True,
                               binarize=False)['data']
    shuffled_predictions[(predictor_key, target_key)] = \
                        run_prediction(DV_scores, 
                                       target, 
                                       results_dir,
                                       outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                       shuffle=shuffle_reps,
                                       classifier=classifier, 
                                       verbose=True,
                                       save=True,
                                       binarize=False)['data']                           

# data
data = {'all_predictions': predictions,
        'all_shuffled_predictions': shuffled_predictions,
        'predictors': predictors,
        'targets': targets}
                        
                        
# save all results
pickle.dump(data, 
            open(path.join(results_dir, 'ideo_predictions.pkl'), 'wb'))
