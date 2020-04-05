import python_speech_features as psf
from sklearn.mixture import GaussianMixture
import joblib
import numpy as np
import os
import scipy.io.wavfile as wav
import speechpy
from collections import Counter
import pickle

class GMM_UBM:

    def __init__(self, no_components=128):

        self.data_path = 'data'
        self.UBM_path = 'data/ubm/'
        self.enroll_path = 'data/enroll/'
        self.test_path = 'data/test/'
        self.no_components=no_components

        if len(os.listdir(self.UBM_path) ) == 0:
            return("UBM directory is empty")
        elif len(os.listdir(self.enroll_path) ) == 0:
            return("Enroll directory is empty")
        elif len(os.listdir(self.test_path) ) == 0:   
            return("Test directory is empty")

        self.list_files_ubm = [self.UBM_path + f for f in os.listdir(self.UBM_path)]
        print("Number of UBM files: ", len(self.list_files_ubm))
        self.list_files_enroll = [f for f in os.listdir(self.enroll_path)]
        print("Number of enrollment speakers: ", len(self.list_files_enroll))
        self.list_files_test = [f for f in os.listdir(self.test_path)]
        print("Number of test files: ", len(self.list_files_test))


    def extract_ubm_mfcc(self):

        mfcc_list = []

        with open('mfcc/ubm_to_mfcc.txt', 'wb') as f:
            print("Extracting UBM MFCCs...")
            for file in self.list_files_ubm:
                fs, signal = wav.read(file)

                mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
                mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True).tolist()

                for mf in mfcc_cmvn:
                    mfcc_list.append(mf)

                #list_mfcc.append(mfcc_cmvn.tolist())

            pickle.dump(mfcc_list, f)
            f.close()
        print("Done")

    def extract_enroll_mfcc(self):

        print("Extracting Enrollment MFCCs...")

        for model in self.list_files_enroll:

            files = [os.path.join(self.enroll_path,model) + "/" + f for f in os.listdir(os.path.join(self.enroll_path, model)) if f[-4:] == '.wav']

            mfcc_list = []

            with open('mfcc/enroll/%s.txt'%(model), 'wb') as f:
                for file in files:
                    
                    fs, signal = wav.read(file)
                    mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
                    mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True).tolist()

                    for mf in mfcc_cmvn:
                        mfcc_list.append(mf)

                pickle.dump(mfcc_list, f)
                f.close()
        print("Done")

    def extract_test_mfcc(self):

        print("Extracting Test MFCCs...")

        for model in self.list_files_test:

            files = [os.path.join(self.test_path,model) + "/" + f for f in os.listdir(os.path.join(self.test_path, model)) if f[-4:] == '.wav']

            mfcc_list = []

            with open('mfcc/test/%s.txt'%(model), 'wb') as f:
                for file in files:

                    fs, signal = wav.read(file)
                    mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                                 num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
                    mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True).tolist()
                    
                    for mf in mfcc_cmvn:
                        mfcc_list.append(mf)

                pickle.dump(np.array(mfcc_list), f)
                f.close()
        print("Done")

    def ubm_model(self):

        gmm = GaussianMixture(n_components= self.no_components, covariance_type= 'diag')
        print("Training UBM...")
        with open('mfcc/ubm_to_mfcc.txt', 'rb') as f:
            file = pickle.load(f)
            X = np.array([np.array(xi) for xi in file])

            gmm_fit = gmm.fit(X)

        with open('dumps/ubm/ubm.pkl', 'wb') as f:
            pickle.dump(gmm_fit, f)

        print("UBM model fitted")

    def enroll_model(self):

        with open('dumps/ubm/ubm.pkl', 'rb') as ubm:
            ubm = pickle.load(ubm)

        for model in self.list_files_enroll:

            file = os.path.join('mfcc/enroll',model) + ".txt"

            with open(file, 'rb') as f:
                f = pickle.load(f)
                X = np.array([np.array(xi) for xi in f])

                gmm = GaussianMixture(n_components= self.no_components, covariance_type= 'diag', means_init=ubm.means_)
                gmm_fit = gmm.fit(X)

                with open('dumps/enroll/%s.pkl'%model, 'wb') as f2:
                    pickle.dump(gmm_fit, f2)

        print("Enrollment Speaker Models fitted")

    def predict(self):

        with open('dumps/ubm/ubm.pkl', 'rb') as ubm:
            ubm = pickle.load(ubm)

        return_dict = {}

        for test in self.list_files_test:

            score_base = 0
            model_pred = 'None'

            file = os.path.join('mfcc/test',test) + ".txt"
            with open(file, 'rb') as mfcc:
                mfcc = pickle.load(mfcc)
                mfcc = np.array([np.array(xi) for xi in mfcc])

            for model in [f for f in os.listdir("dumps/enroll/") if '.pkl' in f]:

                with open(os.path.join("dumps/enroll", model), 'rb') as gmm_enroll:
                    gmm_enroll = pickle.load(gmm_enroll)

                score = gmm_enroll.score(mfcc) - ubm.score(mfcc)
                
                if score < 0:
                    pass

                elif score >= score_base:
                    model_pred = model[:-4]
                    score_base = score

            return_dict[test] = [model_pred, score_base]
        return return_dict

