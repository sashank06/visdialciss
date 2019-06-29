

import torch
from torch import nn
from torch.nn import functional as F

from visdialch.utils import DynamicRNN


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        
        self.dropout = nn.Dropout(p=config["dropout"])

        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        
        # fc layer for image * question to attention weights
        self.Wk = nn.Linear(config["lstm_hidden_size"] , config["lstm_hidden_size"] )
        self.Wq = nn.Linear(config["lstm_hidden_size"] , config["lstm_hidden_size"] )
        self.Wv = nn.Linear(config["lstm_hidden_size"] , config["lstm_hidden_size"] )
        
        
        self.project_attention = nn.Linear(config["lstm_hidden_size"], 1)
        # fusion layer (attended_image_features + question + history)
        fusion_size = (
            config["lstm_hidden_size"] * 2
        )
        self.fusion = nn.Linear(fusion_size, config["lstm_hidden_size"])

        #nn.init.kaiming_uniform_(self.image_features_projection.weight)
        #nn.init.constant_(self.image_features_projection.bias, 0)
        nn.init.kaiming_uniform_(self.fusion.weight)
        nn.init.constant_(self.fusion.bias, 0)

    def forward(self, batch):
        # shape: (batch_size, img_feature_size) - CNN fc7 features
        # shape: (batch_size, num_proposals, img_feature_size) - RCNN features
        img = batch["img_feat"]
        # shape: (batch_size, 10, max_sequence_length)
        ques = batch["ques"]
        # shape: (batch_size, 10, max_sequence_length * 2 * 10)
        # concatenated qa * 10 rounds
        hist = batch["hist"]
        
        # num_rounds = 10, even for test (padded dialog rounds at the end)
        batch_size, num_rounds, max_sequence_length = ques.size()

        # embed questions
        ques = ques.view(batch_size * num_rounds, max_sequence_length)
        ques_embed = self.word_embed(ques)

        # shape: (batch_size * num_rounds, max_sequence_length,
        #         lstm_hidden_size)
        (question_outputs, _), (ques_embed, _) = self.ques_rnn(ques_embed, batch["ques_len"])

        # project down image features and ready for attention
        # shape: (batch_size, num_proposals, lstm_hidden_size)
        # projected_image_features = self.image_features_projection(img)

        # repeat image feature vectors to be provided for every round
        # shape: (batch_size * num_rounds, num_proposals, lstm_hidden_size)

        # embed history
        hist = hist.view(batch_size * num_rounds, max_sequence_length * 20)
        hist_embed = self.word_embed(hist)
        
        # shape: (batch_size * num_rounds, lstm_hidden_size)
        (history_outputs, _), (hist_embed, _) = self.hist_rnn(hist_embed, batch["hist_len"])
        
        
        #Self attention with history 
        
        '''Q = self.Wq(history_outputs)
        K = self.Wk(history_outputs)
        V = self.Wv(history_outputs)

        attention_weights = Q * K
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights * V
        '''
        #Repeat question 
        projected_ques_features = ques_embed.unsqueeze(1).repeat(1, max_sequence_length*20, 1)                                                                                          
        projected_ques_features = projected_ques_features * history_outputs  
        projected_ques_features = self.dropout(projected_ques_features)

        history_attention_weights = self.project_attention(projected_ques_features).squeeze()
        history_attention_weights = F.softmax(history_attention_weights, dim=-1)

        history_attention_weights = history_attention_weights.unsqueeze(-1).repeat(1, 1, self.config["lstm_hidden_size"])
        
        history_attention_weights = (history_attention_weights * history_outputs).sum(1)
        #Fused vector size (batch_size * num_rounds, lstm_hidden_size*3)
        fused_vector = torch.cat((ques_embed, history_attention_weights), 1)
        fused_vector = self.dropout(fused_vector)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        # shape: (batch_size, num_rounds, lstm_hidden_size)
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        
        return fused_embedding , history_attention_weights
