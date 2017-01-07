from expanalysis.experiments.processing import get_battery_DVs
from os import path
import pandas as pd
from selfregulation.utils.utils import get_info

#load Data
token = get_info('expfactory_token')
try:
    data_dir=get_info('data_directory')
except Exception:
    data_dir=path.join(get_info('base_directory'),'Data')

for sample in ['discovery']:
    data = pd.read_json(data_dir + 'mturk_' + sample + '_data_post.json')
    #calculate DVs
    DV_df, valence_df = get_battery_DVs(data)
    DV_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV.json'))
    valence_df.to_json(path.join(data_dir, 'mturk_' + sample + '_DV_valence.json'))