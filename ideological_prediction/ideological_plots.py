import numpy as np
from os import makedirs, path
import pickle

from plot_utils import plot_prediction, plot_prediction_scatter
from selfregulation.utils.plot_utils import save_figure
from selfregulation.utils.utils import  get_info

results_dir = path.join(get_info('results_directory'), 'ideology_prediction')
plot_dir = path.join(results_dir, 'Plots')
makedirs(plot_dir, exist_ok=True)


# load predictions
data = pickle.load(open(path.join(results_dir, 
                                'ideo_predictions.pkl'), 'rb'))
all_predictions = data['all_predictions']
all_shuffled_predictions = data['all_shuffled_predictions']
predictors = data['predictors']
targets = data['targets']

from collections import OrderedDict as odict
predictions = odict()
shuffled_predictions = odict()
for target in targets.keys():
    for predictor_key in ['demographics', 'task', 'survey', 'full_ontology']:
        predictions[predictor_key] = all_predictions[(predictor_key, target)]
        shuffled_predictions[predictor_key] = all_shuffled_predictions[(predictor_key, target)]
        # plot scatter plot
        plot_prediction_scatter(predictions,
                                predictors[predictor_key],
                                targets[target],
                                size=15,
                                filename=path.join(plot_dir, '%s_%s_scatter.png' % (predictor_key, target)))
    plot_prediction(predictions, shuffled_predictions, 
                    target_order=targets[target].columns,
                    filename=path.join(plot_dir, '%s_bar.png' % target))