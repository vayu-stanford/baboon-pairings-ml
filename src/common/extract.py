import numpy as np

DATA_FILE_PATH = '../../data/rawdata.csv'

COLUMN_NAMES = ['female_id', 'male_id', 'cycle_id', 'consort', 'conceptive', 'female_hybridscore', 'male_hybridscore', 'female_gendiv', 'male_gendiv', 'gen_distance', 'female_age', 'male_rank', 'female_rank', 'males_present', 'females_present', 'male_rank_transform', 'gen_distance_transform', 'rank_interact', 'assort_index', 'female_age_transform']
base_features = ['conceptive', 'female_hybridscore', 'male_hybridscore', 'female_gendiv', 'male_gendiv', 'gen_distance', 'female_age', 'male_rank', 'female_rank', 'males_present', 'females_present' ] 
transformed_features = [ 'male_rank_transform', 'gen_distance_transform', 'assort_index', 'female_age_transform']
id_columns = ['female_id', 'male_id', 'cycle_id']
label_columns = ['consort'] 

def generate_labelled_data(valid_labels=None, label_type='consort', include_transformed=True):
    ''' Generate attributes and labels '''
    feature_list = base_features
    if include_transformed:
        feature_list= feature_list + transformed_features
    feature_idxs = get_rawdata_idxs_for_cols(feature_list)
    label_idxs = get_rawdata_idxs_for_cols([label_type])
    print label_idxs
    return extract_labelled_data(feature_idxs, label_idxs, valid_labels)

def extract_labelled_data(feature_idxs, label_idxs, valid_labels):
    attrs=np.loadtxt(DATA_FILE_PATH,delimiter=',',usecols=feature_idxs, skiprows=1)
    labels=np.genfromtxt(DATA_FILE_PATH,delimiter=',',usecols=label_idxs,dtype='str', skip_header=1)
    valid_indices=[i for i in range(labels.shape[0]) if valid_labels == None or labels[i] in valid_labels]
    labels=labels[valid_indices]
    attrs=attrs[valid_indices,:]
    return [attrs, labels]

def get_rawdata_idxs_for_cols(feature_names):
    idxs = []
    for (idx, column) in enumerate(COLUMN_NAMES):
        if column in feature_names:
            idxs.append(idx)
    return idxs
