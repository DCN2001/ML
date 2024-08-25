import os
import numpy as np
import torch
import math
from tqdm import tqdm
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

#With mixing up substems
class MoisesDB(torch.utils.data.Dataset):
    def __init__(self, data, mode):
        self.segment_length = 44100 * 3
        self.hop_size = 44100 * 1
        self.data = data
        self.label_map = {'bass': 0, 'bowed_strings': 1, 'drums': 2, 'guitar': 3, 'wind': 4, 'other_keys': 5, 'other_plucked': 6, 'percussion': 7, 'piano': 8, 'vocals': 9, 'other': 10}
        
        if mode=="train":
            self.song_list = list(self.data.keys())[:200]       #200 songs for training
        elif mode=="test":
            self.song_list = list(self.data.keys())[-40:]       #40 songs for evaluating

        self.song_length = self.est_length(self.song_list)   #Number of segments in each song

        #Build up dataset
        self.dataset = []
        #seg = 0   #if seg >= self.segments_num[song_id]  go to next song
        for song_id in range(len(self.song_list)):
            for seg in range(500):      #Randomly pick 500 starting point
                self.dataset.append(song_id)
                
    def __len__(self):
        return 500 * len(self.song_list)  #sum(self.segments_num)
    
    def est_length(self, song_list):
        song_length = []
        #print("Estimating song length.....")
        for song in song_list:
            instrument_list = self.data[song]
            min_length = math.inf 
            for instrument, instru_wave in instrument_list.items():
                wave_length = instru_wave.shape[1]
                if wave_length < min_length:
                    min_length = wave_length
            song_length.append(min_length)
        return song_length
    
    def __getitem__(self, idx):
        song_id = self.dataset[idx]
        song = self.song_list[song_id]
        stems = []
        labels = []
        #Randomly decide the start point & end point 
        start_pt = np.random.randint(0, self.song_length[song_id]-self.segment_length)
        end_pt = start_pt + self.segment_length
        
        instrument_list = self.data[song]
        for instrument, instru_wave in instrument_list.items():
            track_list = self.data[song][instrument]
            # tracks = list(track_list.values())
            #sliced_tracks = [track[start_pt:end_pt] for track in tracks]
            # #print(song, instrument)
            # instru_wave = np.sum(sliced_tracks, axis=0)
            instru_wave = instru_wave[:,start_pt:end_pt]
            #Check silent or not, if not => add to stem & label it 
            if np.sum(np.abs(instru_wave)) != 0:
                label_encoding = [0] * len(self.label_map)
                label_encoding[self.label_map[instrument]] = 1
                labels.append(np.array(label_encoding, dtype=np.float32)) 
                stems.append(instru_wave)
            else:
                stems.append(np.zeros((2,self.segment_length)))
                labels.append(np.zeros(len(self.label_map)))     
        #Mixture 
        if len(stems) > 0:
            mixture = np.sum(stems, axis=0).astype(np.float32)
        else:
            mixture = np.zeros((self.segment_length,2), dtype=np.float32)

        #label_attractor
        #label_attractor = np.ones((1,len(stems) + 1), dtype=np.float32)
        #label_attractor[:,-1] = 0.0
        #Zeros for decoder
        #zeros = np.zeros((len(stems) + 1,256), dtype=np.float32)
        num_instru = len(stems)
        
        #Collate to make the batch have same size
        if len(stems) < len(self.label_map):
            res = len(self.label_map) - len(stems)
            for r in range(res):
                stems.append(np.zeros((2,self.segment_length),dtype=np.float32))   #np.zero for res instrument
                labels.append(np.zeros((len(self.label_map)),dtype=np.float32))  #np.zero for res instrument
            #label_attractor = np.concatenate([label_attractor,np.full((1,res), 2, dtype=np.float32)], axis=1)
            #zeros = np.vstack([zeros,np.full((res, 256), 2, dtype=np.float32)])
        
        
        return mixture, num_instru, np.array(stems), np.array(labels)  #mixture, label_attractor, wavs of stem, classes of stem
        

def load_data(data, batch_size, n_cpus):
    #Build dataset
    train_ds = MoisesDB(data,"train")
    test_ds = MoisesDB(data, "test")

    #Build dataloader
    train_loader = torch.utils.data.DataLoader(dataset= train_ds,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=n_cpus)
    test_loader = torch.utils.data.DataLoader(dataset= test_ds,
                                              batch_size=batch_size,
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=n_cpus)
    return train_loader, test_loader


'''
#Debugging
datapath = "/home/data1/dcn2001/moisesdb.npy"
print("Loading npy file.......")
data = np.load(datapath, allow_pickle=True).item()
train_loader, eval_loader = load_data(data, 8, 1)
for idx, batch in enumerate(tqdm(train_loader,desc="Train bar",colour="#ff7300")):
    #print(batch[0].shape, len(batch[1]), len(batch[2]))
    continue
'''







