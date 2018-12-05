# remove sklearn deprecation warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# imports
import argparse
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-classifier', default='ridge')
parser.add_argument('-raw_classifier', default='lasso')
parser.add_argument('-EFA_rotation', default='oblimin')
parser.add_argument('-shuffle_repeats', type=int, default=1000)
args = parser.parse_args()

from fancyimpute import SoftImpute
import json
from os import makedirs, path
import numpy as np
import pandas as pd
import pickle
from dimensional_structure.prediction_utils import run_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_recent_dataset, get_info


# parse args
classifier = args.classifier
raw_classifier = args.raw_classifier
shuffle_reps = args.shuffle_repeats
EFA_rotation = args.EFA_rotation

# load data
dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv')
results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
makedirs(results_dir, exist_ok=True)

# run prediction
shuffle_reps = 1000

# define predictors
survey_scores = results['survey'].EFA.get_scores(rotate=EFA_rotation)
task_scores = results['task'].EFA.get_scores(rotate=EFA_rotation)
predictors = {'survey_%s' % EFA_rotation: survey_scores,
              'task_%s' % EFA_rotation: task_scores,
              'demographics': results['task'].DA.get_scores(),
              'full_ontology_%s' % EFA_rotation: pd.concat([survey_scores, task_scores], axis=1)}
# define targets
targets = {'ideo_factors': ideo_data.filter(regex='Factor'),
           'ideo_orientations': ideo_data.drop(ideo_data.filter(regex='Factor|SECS').columns, axis=1).drop(['Conservatism','Intellectual Humility'], axis=1),
           'ideo_policies': ideo_data.filter(regex='SECS')}
for key, target in targets.items():
    imputed = pd.DataFrame(SoftImpute().complete(target),
                            index=target.index,
                            columns=target.columns)
    targets[key] = imputed+1E-5

# do prediction with ontological factors
predictions = {}
shuffled_predictions = {}
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
for target_key, target in targets.items():
    print('Running Prediction: raw measures predicting %s' % target_key)
    predictions[(predictor_key, target_key)] = \
                run_prediction(DV_scores, 
                               target, 
                               results_dir,
                               outfile='%s_%s_prediction' % (predictor_key, target_key), 
                               shuffle=False,
                               classifier=raw_classifier, 
                               verbose=True,
                               save=True,
                               binarize=False)['data']
    shuffled_predictions[(predictor_key, target_key)] = \
                        run_prediction(DV_scores, 
                                       target, 
                                       results_dir,
                                       outfile='%s_%s_prediction' % (predictor_key, target_key), 
                                       shuffle=shuffle_reps,
                                       classifier=raw_classifier, 
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
            open(path.join(results_dir, 
                           'ideo_predictions_%s.pkl' % EFA_rotation), 'wb'))

# save results in easier-to-use format
simplified = {}
simplified_importances = {}
predictor_importances = {}
for p in predictors.keys():
    simplified[p] = {}
    simplified
    predictor_importances[p] = {}
    for t in targets.keys():
        tmp = predictions[(p,t)]
        tmp_scores = {'CV_'+k:tmp[k]['scores_cv'][0]['R2'] for k in tmp.keys()}
        tmp_scores.update({'insample_'+k:tmp[k]['scores_insample'][0]['R2'] for k in tmp.keys()})
        simplified[p].update(tmp_scores)
        # get importances
        for k in tmp.keys():
            importances = tmp[k]['importances'][0]
            predvars = tmp[k]['predvars']
            non_zero = np.where(importances)[0]
            zipped = list(zip([predvars[i] for i in non_zero],
                                           importances[non_zero]))
            predictor_importances[p][k] = sorted(zipped, 
                                                 key = lambda x: abs(x[1]), 
                                                 reverse=True)
simplified['Target_Cat'] = {}
for t,vals in targets.items():
    simplified['Target_Cat'].update({'CV_' + c:t for c in vals.columns})
    simplified['Target_Cat'].update({'insample_' + c:t for c in vals.columns})
        
simplified=pd.DataFrame(simplified)
simplified.to_csv(path.join(results_dir, '
                            predictions_%s_R2.csv' % EFA_rotation))
json.dump(predictor_importances, 
          open(path.join(results_dir, 
                         'predictor_importances_%s.json' % EFA_rotation), 'w'))






