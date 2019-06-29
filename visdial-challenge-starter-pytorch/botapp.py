import time
import random
import telepot
import os
import h5py
from nltk import word_tokenize
import argparse
import json
import os
import re

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from visdialch.data.dataset import VisDialDataset
from visdialch.encoders.lf import LateFusionEncoder as Encoder
from visdialch.decoders import Decoder
from visdialch.metrics import SparseGTMetrics, NDCG, scores_to_ranks
from visdialch.model import EncoderDecoderModel
from visdialch.utils.checkpointing import load_checkpoint

import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
import json
import pickle

available_image_ids = {
    float(re.findall(r'([\d]+)\.jpg', f)[0])
    :
    os.path.join(f'../visdial-challenge-starter-pytorch/data/images/{split}/{f}')
    for split in (
        'train2014',
        'test2014',
        'val2014',
        'VisualDialog_test2018',
        'VisualDialog_val2018',
    )
    for f in os.listdir(f'../visdial-challenge-starter-pytorch/data/images/{split}/')
    }

# keys: {"dataset", "model", "solver"}
config = yaml.load(open('checkpoints/new_features_baseline/config.yml'))

val_dataset = VisDialDataset(
    config["dataset"],
    "data/visdial_1.0_val.json",
    "data/visdial_1.0_val_dense_annotations.json",
    return_options=True,
    add_boundary_toks=False
    if config["model"]["decoder"] == "disc"
    else True,
)

# Pass vocabulary to construct Embedding layer.
encoder = Encoder(config["model"], val_dataset.vocabulary)
decoder = Decoder(config["model"], val_dataset.vocabulary)
print("Encoder: {}".format(config["model"]["encoder"]))
print("Decoder: {}".format(config["model"]["decoder"]))

model = EncoderDecoderModel(encoder, decoder)
model_state_dict, _ = load_checkpoint('checkpoints/new_features_baseline/checkpoint_10.pth')
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(model_state_dict)
else:
    model.load_state_dict(model_state_dict)
model.eval()

with open('data/val_data.pkl','rb') as file:
    (img_ids, caption_vectors, all_captions, all_questions, all_questions_vectors,
    all_answers, all_questions) = pickle.load(file)
    
def jon(query, questions, answers, image_id):
    index = img_ids.index(18472.0)
    caption = all_captions[index]
    
    data_dict = {
        "version":"1.0",
        "split":"test",
        "data":{
            "dialogs":[{
                'image_id':image_id,
                'dialog':[],
                'caption':caption
            }],
            "answers":[],
            "questions":[]
        }
    }
    
    query_tokenized = word_tokenize(query)
    query_to_ix = val_dataset.vocabulary.to_indices(query_tokenized)
    
    query_embedding = model.encoder.word_embed(torch.tensor(query_to_ix)).mean(0).detach().numpy()
    
    query_distances = pairwise_distances(
        [np.array(query_embedding)],
        [q[2] for q in all_questions], metric='cosine'
    )
    
    most_similar_question = all_questions[query_distances.argmin()]
    (
        question_str,
        question_numpy,
        question_embedding,
        answers_numpy,
        answers_tensor_len) = most_similar_question
    
    possible_answers_str=list()
    for a in answers_numpy:
        possible_answers_str.append(
            ' '.join([item for item in val_dataset.vocabulary.to_words(a) if item !='<PAD>']))
    
    data_dict['data']['questions'] = ['']+[query]+questions
    data_dict['data']['answers'] = ['']+answers+possible_answers_str
    
    # history
    for e, (q, a) in enumerate(zip(questions, answers)):
        data_dict['data']['dialogs'][0]['dialog'].append({
            'answer':e+1,
            'question':2+e,
            'answer_options':list(range(
                len(data_dict['data']['answers'])-100,
                len(data_dict['data']['answers']))
            ),
            'gt_index':0
        })
    data_dict['data']['dialogs'][0]['dialog'].append({
        'answer':0,
        'question':1,
        'answer_options':list(range(
                len(data_dict['data']['answers'])-100,
                len(data_dict['data']['answers']))
            ),
        'gt_index':0
    })
    
    dialogue_len=len(data_dict['data']['dialogs'][0]['dialog'])
    data_dict['possible_answers']=possible_answers_str
    return data_dict

config['dataset']['image_features_train_h5']='data/features_thomas_val.h5'
config['dataset']['image_features_test_h5']='data/features_thomas_val.h5'
config['dataset']['image_features_val_h5']='data/features_thomas_val.h5'


with open('data/val_bot_data.pkl', 'rb') as file:
    all_image_ids, all_captions, all_caption_similarities = pickle.load(file)

class TelegramBot():
    def __init__(self, token, chat_id):
        self._token = token
        self._chat_id = chat_id
        self.bot = telepot.Bot(token)
        
        self.received_history = list()
        self.sent_history = list()
        self.history = list()
        
        self.received_message_ids = self.check_received_message_ids()
    
    def start_game(self):
        #todo
        self.questions = list()
        self.answers = list()
        
        
        self.correct_image_ix = np.random.choice(range(len(img_ids)))
        self.correct_img_id = all_image_ids[self.correct_image_ix]
        self.correct_caption = all_captions[self.correct_image_ix]
        
        self.send_msg(str(self.correct_image_ix)+' '+str(self.correct_img_id)+  ' '+str(self.correct_caption))
        
        similarities_vec = all_caption_similarities[self.correct_image_ix].copy()
        other_images_indices = similarities_vec.argsort()[-3:][::-1]
        other_images_ids = np.array(all_image_ids)[other_images_indices]
        other_captions = np.array(all_captions)[other_images_indices]
        
        curr_images_ids = np.array([self.correct_img_id]+list(other_images_ids))
        curr_captions = np.array([self.correct_caption]+list(other_captions))
        shuffle_indices = np.random.permutation(range(0, 4))
        self.send_msg('shuffle_indices '+str(shuffle_indices))
        
        shuffled_images_ids = curr_images_ids[shuffle_indices]
        self.send_msg('shuffled_images_ids '+str(shuffled_images_ids))
        
        shuffled_captions = curr_captions[shuffle_indices]
        self.send_msg('shuffled_captions '+str(shuffled_captions))
        
        
        self.correct_image_position = np.where(shuffle_indices==0)[0][0]
        for im_id, cap in zip(shuffled_images_ids, shuffled_captions):
            self.send_pic(available_image_ids[im_id], caption=cap+f'({str(im_id)})')
    
    def chat(self, query):
        if len(self.questions)==10:
            self.send_msg('You ran out of question... Please predict with /predict <number_of_image> (1-4)')
            return
        J = jon(query=query, answers=self.answers, questions=self.questions,image_id=self.correct_img_id)
        
        with open('/tmp/J.json','w') as file:
            json.dump(J, file)
        val_dataset = VisDialDataset(
            config["dataset"],
            "/tmp/J.json",
            "hello",
            return_options=True,
            add_boundary_toks=False
            if config["model"]["decoder"] == "disc"
            else True,
        )
        sample = val_dataset[0]

        forward=model({k: val.unsqueeze(0) for k, val in sample.items()})
        
        #self.send_msg('forward.shape'+str(forward.shape))
        best_answer_index = int(forward[0][0][len(self.questions)].argmax())
        with open('/tmp/J.json','r') as file:
            best_answer = json.load(file)['possible_answers'][best_answer_index]
        self.questions.append(query)
        self.answers.append(best_answer)
        self.send_msg(best_answer)
    
    def predict(self, query):
        #/predict 1
        score = query[9:]
        try:
            if int(self.correct_image_position) == int(score)-1:
                self.send_msg('CONGRATZ🎉')
            else:
                self.send_msg('You FAILED👎')
            self.send_msg(f'Correct is {self.correct_img_id}')
        except:
            self.send_msg('Invalid /predict syntax.')
        
    def check_received_message_ids(self):
        out = list()
        for update in self.bot.getUpdates():
            try:
                out.append(update['message']['message_id'])
            except:
                pass
        return out
    
    def check_last_message_and_maybe_do_action(self):
        last_msg = self.bot.getUpdates()[-1]['message']['text']
        if last_msg=='/start':
            self.start_game()
        elif last_msg=='/predict':
            #todo
            self.predict()
        elif last_msg=='/rand_pic':
            self.send_random_pic()
        elif last_msg.startswith('/repeat '):
            self.send_msg(self.bot.getUpdates()[-1]['message']['text'][8:])
        elif last_msg.startswith('/predict'):
            self.predict(query=last_msg)
        # Fallback case, playing the game
        else:
            self.chat(query=last_msg)
        
    def repeat_last_received_message(self):
        self.send_msg(self.bot.getUpdates()[-1]['message']['text'])
        
    def send_msg(self, msg):
        self.bot.sendMessage(chat_id=self._chat_id, text=msg)
        self.sent_history.append(msg)
    
    def send_pic(self, pic_path, caption=None):
        with open(pic_path, 'rb') as file:
            self.bot.sendPhoto(self._chat_id, file, caption=caption)
    
    def send_random_pic(self):
        available_pictures = os.listdir('data/images/train2014/')
        random.shuffle(available_pictures)
        random_pic_path = os.path.join('data/images/train2014/', available_pictures[0])
        self.send_pic(pic_path=random_pic_path, caption=random_pic_path)
        
    def flush(self):
        self.received_history = list()
        self.sent_history = list()
        self.history = list()
        self.send_msg("I'm flushed -> 🗑️")
    
def main():
    bot = TelegramBot(
        token="837927537:AAH84XUk7HYoKl0CD2WLoM_UpluiSsWnDi8",
        chat_id=524089443
    )
    bot.send_msg("I'm up!")
    while True:
        if bot.check_received_message_ids() != bot.received_message_ids:
            bot.received_message_ids = bot.check_received_message_ids()
            #bot.repeat_last_received_message()
            bot.check_last_message_and_maybe_do_action()
            
        time.sleep(1)
    
    
if __name__=='__main__':
    main()