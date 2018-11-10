
from dimensional_structure.prediction_utils import run_prediction
from selfregulation.utils.result_utils import load_results
from selfregulation.utils.utils import get_behav_data, get_demographics, get_recent_dataset

dataset = get_recent_dataset()
results = load_results(dataset)
ideo_data = get_behav_data(dataset, file='ideology.csv')
demo = get_demographics(dataset=dataset)
classifiers = ['lasso', 'ridge',  'svm', 'rf']

# run prediction
target_name = 'ideology'
target = ideo_data+1E-10 # adding a small float causes values to be seen as continuous by "type_of_target"
shuffle = False
classifier = classifiers[1]
predictors = ['survey', results['survey'].EFA.get_scores()]

out = run_prediction(predictors[1], 
               target, 
               '/home/ian/tmp',
               outfile='%s_%s_prediction' % (predictors[0], target_name), 
               shuffle=shuffle,
               classifier=classifier, 
               verbose=True,
               save=False)