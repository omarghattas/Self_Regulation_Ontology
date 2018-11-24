from collections import OrderedDict as odict
from os import makedirs, path
import pandas as pd
import pickle
from ideological_prediction.plot_utils import (plot_outcome_ontological_similarity,
                                               plot_prediction, plot_prediction_scatter,
                                               polar_plots,
                                               plot_predictors_comparison)
from selfregulation.utils.utils import  get_info

results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
plot_dir = path.join(results_dir, 'Plots')
makedirs(plot_dir, exist_ok=True)


# load predictions
ext = 'pdf'
data = pickle.load(open(path.join(results_dir, 
                                'ideo_predictions.pkl'), 'rb'))
all_predictions = data['all_predictions']
all_shuffled_predictions = data['all_shuffled_predictions']
predictors = data['predictors']
targets = data['targets']

for target in targets.keys():
    predictions = odict()
    shuffled_predictions = odict()
    for predictor_key in ['demographics', 'task', 'survey', 'full_ontology', 'raw_measures']:
        predictions[predictor_key] = all_predictions[(predictor_key, target)]
        shuffled_predictions[predictor_key] = all_shuffled_predictions[(predictor_key, target)]
        # plot scatter plot
        plot_prediction_scatter(predictions[predictor_key],
            predictors[predictor_key],
            targets[target],
            size=15,
            filename=path.join(plot_dir, '%s_%s_scatter.%s' % (predictor_key, target, ext)))
        # plot ontological similarity
        if predictor_key != 'raw_measures':
            plot_outcome_ontological_similarity(predictions[predictor_key],
                            size=15,
                            filename=path.join(plot_dir, '%s_%s_similarity.%s' % (predictor_key, target, ext)))
    plot_prediction(predictions, shuffled_predictions, 
                    target_order=targets[target].columns,
                    filename=path.join(plot_dir, '%s_bar.%s' % (target, ext)))
    # drop some unneeded predictions
    for key in ['demographics', 'raw_measures']:
        del predictions[key]
    if target == 'ideo_factors':
        polar_plots(predictions, target_order=targets[target].columns,
                    size=10,
                    filename=path.join(plot_dir, '%s_polar_plots.%s' % (target,ext)))
        
# compare performance as distribution
simplified = pd.read_csv(path.join(results_dir, 'predictions_R2.csv'), 
                         index_col=0)
plot_predictors_comparison(simplified,
                           size=3,
                           filename=path.join(plot_dir, 'prediction_compare.pdf'))