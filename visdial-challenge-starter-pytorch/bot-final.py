import telegram
from telegram.ext import Updater
from telegram.ext import CommandHandler, Handler
from telegram.ext import MessageHandler, Filters
import logging
import telepot
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
import matplotlib.pyplot as plt
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
import code
import pickle

TOKEN="837927537:AAH84XUk7HYoKl0CD2WLoM_UpluiSsWnDi8"
CHAT_ID="524089443"


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
    
    def explain(self):
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

        _, attention_weights =model({k: val.unsqueeze(0) for k, val in sample.items()})
        attention_weights = attention_weights.detach().cpu().numpy()
        
        pil_image = Image.open(available_image_ids[self.correct_img_id])
        w, h = pil_image.size
        
        for ix_ques, (ques, ans) in enumerate(zip(self.questions, self.answers)):
            weights = attention_weights[ix_ques]
            weights = np.resize(weights, (8, 8))
            
            fig = plt.figure(frameon=False)
            fig.set_size_inches(w/100,h/100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            img = ax.imshow(np.array(pil_image))
            ax.imshow(weights, cmap='gray', alpha=0.6, extent=img.get_extent())
            fig.savefig('/tmp/pic.png')
            self.send_pic('/tmp/pic.png', caption='{}({})'.format(ques, ans))
        
        

        
    def predict(self, prediction):
        #/predict 1
        if int(self.correct_image_position) == int(prediction)-1:
            self.send_msg('CONGRATZ🎉')
        else:
            self.send_msg('You FAILED👎')
        self.send_msg(f'Correct is {self.correct_img_id}')
    
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
    
def main():
    bot = TelegramBot(token=TOKEN,chat_id=CHAT_ID)
    bot.send_msg("I'm up!")
    
    telegram_bot = telegram.Bot(token=TOKEN)
    
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    def start(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        bot.__init__(token=TOKEN, chat_id=CHAT_ID)
        context.bot.send_message(chat_id=update.message.chat_id, text="I'm a bot, please talk to me!")

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    def repeat(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        text = ' '.join(context.args)
        context.bot.send_message(chat_id=update.message.chat_id, text=text)

    repeat_handler = CommandHandler('repeat', repeat)
    dispatcher.add_handler(repeat_handler)

    def play(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        #bot.send_msg('update_chat_id'+str(update_chat_id))
        #bot.send_msg('type update_chat_id'+str(type(update_chat_id)))
        #bot.send_msg('type thomas chat id'+str(type(CHAT_ID)))
        #bot.send_msg('thomas chat id'+str(CHAT_ID))
        
        bot.start_game()
    play_handler = CommandHandler('play', play)
    dispatcher.add_handler(play_handler)
    
    def explain(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        bot.explain()
    explain_handler = CommandHandler('explain', explain)
    dispatcher.add_handler(explain_handler)
    
    def randpic(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        bot.send_random_pic()
    randpic_handler = CommandHandler('randpic', randpic)
    dispatcher.add_handler(randpic_handler)

    def predict(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        try:
            prediction = int(context.args[0])
            bot.predict(prediction)
        except:
            context.bot.send_message(
                chat_id=update.message.chat_id,
                text='Bad /prediction formatting')
            return

    predict_handler = CommandHandler('predict', predict)
    dispatcher.add_handler(predict_handler)

    # Okay, now let's respond to a user input command "caps"
    
    def process_text(update, context):
        update_chat_id = update.message.chat_id
        if int(update_chat_id) != int(CHAT_ID):
            return
        message = update.message.text
        bot.chat(query=message)

    # Filters.text allows the handler to only respond to text messages
    msg_handler = MessageHandler(Filters.text, process_text)
    dispatcher.add_handler(msg_handler)

    updater.start_polling()
    
    
    
    
if __name__=='__main__':
    main()