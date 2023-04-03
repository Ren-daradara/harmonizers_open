import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class CVAE(nn.Module):
    def __init__(self,
                 model_type,
                 device=None
                 ):
        super(CVAE, self).__init__()
        
        self.model_type = model_type
        self.params_prenet_size = 256
        self.params_prenet_num_layers=2
        self.params_encoder_num_layers=2
        self.params_decoder_num_layers=2
        self.params_num_chord=73
        self.params_output_num_chord=72
        self.params_encoder_hidden_size=512
        self.params_decoder_hidden_size=512
        self.params_latent_size=64
        self.BEAT_RESOLUTION=8
        self.max_sequence_length=16
        self.device = device
        
        if self.model_type == 'CVAE':
            self.params_prenet_size = 0 
        
        elif self.model_type == 'SurpriseNet':
            # Surprisingness to prenet
            self.surprise_prenet = nn.LSTM(input_size=1, 
                                   hidden_size = self.params_prenet_size , 
                                   num_layers=self.params_prenet_num_layers,
                                   batch_first=True, 
                                   dropout=0.2,
                                   bidirectional=True)
        
        # Encoder
        self.encoder = nn.LSTM(input_size=self.params_num_chord + self.BEAT_RESOLUTION * 25 + self.params_prenet_size * 2, 
                               hidden_size=self.params_encoder_hidden_size , 
                               num_layers=self.params_encoder_num_layers ,
                               batch_first=True,
                               dropout=0.2,
                               bidirectional=True)
        
        # Encoder to latent
        self.encoder_output2mean = nn.Linear(self.params_encoder_hidden_size * 2, self.params_latent_size,device=self.device)
        self.encoder_output2logv = nn.Linear(self.params_encoder_hidden_size * 2, self.params_latent_size,device=self.device)
        
        # Latent to decoder
        self.latent2decoder_input = nn.Linear(self.params_latent_size + self.BEAT_RESOLUTION * 25 + self.params_prenet_size * 2, self.params_decoder_hidden_size // 2,device=self.device)
        
        # Decoder
        self.decoder = nn.LSTM(input_size=self.params_decoder_hidden_size // 2, 
                               hidden_size =self.params_decoder_hidden_size, 
                               num_layers=self.params_decoder_num_layers, 
                               batch_first=True, 
                               dropout=0.2, 
                               bidirectional=True)
        
        # Decoder to reconstructed chords
        self.outputs2chord = nn.Linear(self.params_decoder_hidden_size * 2,self.params_output_num_chord,device=self.device)

    def surprise_embedding(self, surprise_condition, length):
        
#         print('surprise',surprise_condition.shape)
#         print('length',length.shape)
        # Pack data to encoder
        #packed_x = pack_padded_sequence(surprise_condition, length, batch_first=True, enforce_sorted=False)
        prenet_output , (hidden, _) = self.surprise_prenet(surprise_condition)
        # Pad back
        #prenet_output, _ = pad_packed_sequence(prenet_output, batch_first=True, total_length=self.max_sequence_length)

        return prenet_output
    
    def encode(self, input_seq, length):
        
#         print('input_seq',input_seq.shape)
#         print('length',length.shape)
        # Pack data to encoder
        #packed_x = pack_padded_sequence(input_seq, length, batch_first=True, enforce_sorted=False)
        encoder_output , (hidden, _) = self.encoder(input_seq)
        
        # Pad back
        #encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True, total_length=Constants.MAX_SEQUENCE_LENGTH)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.encoder_output2mean(encoder_output)
        log_var = self.encoder_output2logv(encoder_output)

        return mu, log_var
    
    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
      
        # If cuda
        if torch.cuda.is_available():
            eps = eps.to(self.device)
            
        z = eps * std + mu
        
        return z
      
    def decode(self, z, decoder_input):
        
        z = torch.cat([z, decoder_input],dim=-1)
        
        # Latent to hidden 
        decoder_input = self.latent2decoder_input(z)
        decoder_output, _ = self.decoder(decoder_input)
        
        # Reconstruct to one-hot chord
        result = self.outputs2chord(decoder_output)
        
        # Softmax
        softmax = F.softmax(result,dim=-1)
        
        return result, softmax
    
    def forward(self, input_chord,length,melody_condition,surprise_condition):
        
        if self.model_type == 'SurpriseNet':
            
            # Surprsie contour to prenet
            surprise_condition = self.surprise_embedding(surprise_condition,length)
            melody_condition=torch.reshape(melody_condition,(64,8,-1))
            encoder_input = torch.cat([input_chord,melody_condition,surprise_condition], dim = -1)
            decoder_input = torch.cat([melody_condition, surprise_condition],dim = -1)
        
        elif self.model_type == 'CVAE':
        
            encoder_input = torch.cat([input_chord,melody_condition], dim = -1)
            decoder_input = melody_condition
            
        else: 
            raise NameError('No model name')
            
        # Encode
        mu, log_var = self.encode(encoder_input, length)

        # Reparameterize
        z = self.reparameterize(mu, log_var)

        # Decode
        output, softmax = self.decode(z, decoder_input)
        
        # Log Softmax
        logp = F.log_softmax(output, dim = -1)
    
        return softmax, logp, mu, log_var, input_chord

    def test(self, length,melody_condition,surprise_condition,z):
        surprise_condition = self.surprise_embedding(surprise_condition,length)
        melody_condition=torch.reshape(melody_condition,(1,8,-1))
        decoder_input = torch.cat([melody_condition, surprise_condition],dim = -1)
        Z=z.view(1,8,64)
        print(Z.size())
        output, softmax = self.decode(Z, decoder_input)
        
        # Log Softmax
        logp = F.log_softmax(output, dim = -1)
    
        return softmax, logp