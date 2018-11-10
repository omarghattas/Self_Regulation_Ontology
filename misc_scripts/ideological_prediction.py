from fancyimpute import SoftImpute
import pandas as pd

from dimensional_structure.utils import hierarchical_cluster
from dimensional_structure.prediction_utils import run_prediction
from dimensional_structure.prediction_plots import plot_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_demographics, get_recent_dataset

dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv').drop('WCST_Accuracy', axis=1)
demo = get_demographics(dataset=dataset)
classifiers = ['lasso', 'ridge',  'svm', 'rf']

# cluster ideo_data to order
imputed_ideo = pd.DataFrame(SoftImpute().complete(ideo_data),
                            index=ideo_data.index,
                            columns=ideo_data.columns)
clustering = hierarchical_cluster(imputed_ideo.T, method='average',
                                  pdist_kws={'metric': 'abscorrelation'})
clustered_df = clustering['clustered_df']
# run prediction
target_name = 'ideology'
target = ideo_data+1E-10 # adding a small float causes values to be seen as continuous by "type_of_target"
shuffle = False
classifier = classifiers[1]
predictors = {'survey': results['survey'].EFA.get_scores(),
              'task': results['task'].EFA.get_scores()}

predictions = {}
shuffled_predictions = {}
for key, scores in predictors.items():
    predictions[key] = run_prediction(scores, 
                            target, 
                            '/home/ian/tmp',
                            outfile='%s_%s_prediction' % (key, target_name), 
                            shuffle=shuffle,
                            classifier=classifier, 
                            verbose=True,
                            save=False,
                            binarize=False)['data']
    shuffled_predictions[key] = run_prediction(scores, 
                                    target, 
                                    '/home/ian/tmp',
                                    outfile='%s_%s_prediction' % (key, target_name), 
                                    shuffle=2,
                                    classifier=classifier, 
                                    verbose=True,
                                    save=False,
                                    binarize=False)['data']

plot_prediction(predictions['task'], shuffled_predictions['task'], 
                EFA=results['task'].EFA, target_order=clustered_df.columns,
                size=15, filename='/home/ian/tmp/task_ideo_prediction')

plot_prediction(predictions['survey'], shuffled_predictions['survey'], 
                EFA=results['survey'].EFA, target_order=clustered_df.columns,
                size=15, filename='/home/ian/tmp/survey_ideo_prediction')

