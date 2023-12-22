from os.path import join
from easydict import EasyDict as edict

hist_length = 25
pred_length = 25
total_length = hist_length + pred_length

dim = 63
motion_mlp = edict()
motion_mlp.hidden_dim = dim
motion_mlp.seq_len = hist_length
motion_mlp.with_normalization = True
motion_mlp.spatial_fc_only = False
motion_mlp.norm_axis = 'spatial'

motion_mlp.num_layers = 48

DATASET_FOLDERS = {
                    'H36M':  './dataset/data_processed/h36m_train_%d.pkl' % total_length,
                    'AMASS':  './dataset/data_processed/amass_train_%d.pkl' % total_length,
                    'PW3D':  './dataset/data_processed/pw3d_test_%d.pkl' % total_length,
                                  }

DATASET_VALID_PATH = {
                    # 'AMASS':  './dataset/data_processed/amass_train_%d_sub.pkl' % total_length,
                  }

DATASET_FOLDERS_TEST = {
                    'H36M': './dataset/data_processed/h36m_test_%d.pkl' % total_length,
                    'PW3D':  './dataset/data_processed/pw3d_test_%d.pkl' % total_length,
                    'AMASS':  './dataset/data_processed/amass_test_%d.pkl' % total_length,
                }

partition = [1.]

SMPL_MODEL_PATH = './dataset/smpl_official/processed_basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
SMPLH_N_PATH = './dataset/smpl_official/neutral/model.npz'
SMPLH_M_PATH = './dataset/smpl_official/male/model.npz'
SMPLH_F_PATH = './dataset/smpl_official/female/model.npz'

test_mode = 'AMASS' # 'H36M' 



