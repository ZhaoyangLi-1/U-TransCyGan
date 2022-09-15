import torch
import torch.nn as nn
from einops import rearrange
from models.volume_embedding import Embeddings3D
from models.transformer import TransformerEncoder, TransformerEncoderLayer
from models.transformer import TransformerDecoder, TransformerDecoderLayer
from models.function import calc_mean_std, normal, normal_style


class TransConv3DBlock(nn.Module):
    def __init__(self, embed_dim, base_filters, output_dim, norm):
        super().__init__()
        self.tranCon3D = nn.Sequential(
            nn.ConvTranspose3d(embed_dim, base_filters * 8, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=False),
            nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2,
                               padding=0, output_padding=0, bias=False),
            Conv3DBlock(base_filters, base_filters, double=True, norm=norm),
            nn.Conv3d(base_filters, output_dim, kernel_size=1, stride=1)
        )

    def forward(self, x):
        y = self.tranCon3D(x)
        return y


class ModalityTransTransformer(nn.Module):
    def __init__(self, resNet, input_dim=1, output_dim=1, embed_dim=768,
                 image_shape=(96, 128, 96), patch_size=16, dropout=0.1,
                 num_heads=12, norm='instance', dim_linear_block=2048,
                 num_encoder_layers=12, num_decoder_layers=3, ext_layers=[3, 6, 9, 12],
                 base_filters=16):
        super(ModalityTransTransformer, self).__init__()
        self.resNet = resNet
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.image_shape = image_shape
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.dim_linear_block = dim_linear_block
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.patch_dim = [int(x / patch_size) for x in image_shape]
        # self.squeezeNet = squeezeNet

        # Create Embedding 3D
        self.mse_loss = nn.MSELoss()
        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d
        self.embedding = Embeddings3D(
            input_dim, embed_dim, image_shape, patch_size, dropout)
        # Create Content Encoder
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.encoder_layer_content = TransformerEncoderLayer(d_model=self.embed_dim,
                                                             dim_feedforward=self.dim_linear_block,
                                                             nhead=self.num_heads,
                                                             dropout=self.dropout)
        self.encoder_content = TransformerEncoder(encoder_layer=self.encoder_layer_content,
                                                  num_layers=self.num_encoder_layers, norm=self.encoder_norm,
                                                  ext_layers=self.ext_layers)

        # Create Style Encoder
        self.encoder_layer_style = TransformerEncoderLayer(d_model=self.embed_dim,
                                                           dim_feedforward=self.dim_linear_block,
                                                           nhead=self.num_heads,
                                                           dropout=self.dropout)
        self.encoder_style = TransformerEncoder(encoder_layer=self.encoder_layer_style,
                                                num_layers=self.num_encoder_layers, norm=self.encoder_norm,
                                                ext_layers=self.ext_layers)

        # Create Decoder
        self.decoder_layer = TransformerDecoderLayer(d_model=self.embed_dim,
                                                     dim_feedforward=self.dim_linear_block,
                                                     nhead=self.num_heads,
                                                     dropout=self.dropout)

        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder3 = TransformerDecoder(decoder_layer=self.decoder_layer,
                                           num_layers=self.num_decoder_layers, norm=self.decoder_norm)
        self.decoder6 = TransformerDecoder(decoder_layer=self.decoder_layer,
                                           num_layers=self.num_decoder_layers, norm=self.decoder_norm)
        self.decoder9 = TransformerDecoder(decoder_layer=self.decoder_layer,
                                           num_layers=self.num_decoder_layers, norm=self.decoder_norm)
        self.decoder12 = TransformerDecoder(decoder_layer=self.decoder_layer,
                                            num_layers=self.num_decoder_layers, norm=self.decoder_norm)
        self.seq_sum_linear = nn.Linear(4, 1)
        self.tranCon3d = TransConv3DBlock(
            embed_dim, base_filters, output_dim, self.norm)

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
            self.mse_loss(input_std, target_std)

    def forward(self, content_input, style_input, model):
        if model is 'train':
            content_emd_input = self.embedding(content_input)
            style_emd_input = self.embedding(style_input)

            content_z3, content_z6, content_z9, content_z12 = map(lambda t: rearrange(t, 'b (x y z) d -> b d x y z',
                                                                                      x=self.patch_dim[0],
                                                                                      y=self.patch_dim[1],
                                                                                      z=self.patch_dim[2]),
                                                                  self.encoder_content(content_emd_input))

            style_z3, style_z6, style_z9, style_z12 = map(lambda t: rearrange(t, 'b (x y z) d -> b d x y z',
                                                                              x=self.patch_dim[0],
                                                                              y=self.patch_dim[1],
                                                                              z=self.patch_dim[2]),
                                                          self.encoder_style(style_emd_input))

            content_z3 = rearrange(
                content_z3, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            content_z6 = rearrange(
                content_z6, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            content_z9 = rearrange(
                content_z9, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            content_z12 = rearrange(
                content_z12, 'b d x y z -> b d (x y z)').permute(2, 0, 1)

            style_z3 = rearrange(
                style_z3, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            style_z6 = rearrange(
                style_z6, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            style_z9 = rearrange(
                style_z9, 'b d x y z -> b d (x y z)').permute(2, 0, 1)
            style_z12 = rearrange(
                style_z12, 'b d x y z -> b d (x y z)').permute(2, 0, 1)

            seq3 = self.decoder3(style_z3, content_z3)
            seq6 = self.decoder6(style_z6, content_z6)
            seq9 = self.decoder9(style_z9, content_z9)
            seq12 = self.decoder12(style_z12, content_z12)
            mix_seq = torch.cat([seq3, seq6, seq9, seq12],
                                dim=0).permute(1, 2, 3, 0)
            mix_seq = torch.squeeze(self.seq_sum_linear(
                mix_seq).permute(1, 0, 2, 3), dim=3)
            mix_seq = rearrange(mix_seq, 'b (x y z) d -> b d x y z', x=self.patch_dim[0],
                                y=self.patch_dim[1],
                                z=self.patch_dim[2])

            # change the sequence to output image
            I_o = self.tranCon3d(mix_seq)
            """
            Io_feats = self.resNet(I_o)
            Ic_feats = self.resNet(content_input)
            Is_feats = self.resNet(style_input)
            #print("Io_fears: {}".format(Io_feats.size()))
            #print("Ic_fears: {}".format(Ic_feats.size()))
            loss_c = self.calc_content_loss(normal(Io_feats[-1]), normal(Ic_feats[-1])) + \
                    self.calc_content_loss(normal(Io_feats[-2]), normal(Ic_feats[-2]))
            # Style loss
            loss_s = self.calc_style_loss(Io_feats[-1], Is_feats[-1]) + \
                    self.calc_style_loss(Io_feats[-2], Is_feats[-2])
            """
            loss_c = self.calc_content_loss(normal(I_o[-1]), normal(content_input[-1])) + \
                self.calc_content_loss(
                    normal(I_o[-2]), normal(content_input[-2]))
            # Style loss
            loss_s = self.calc_style_loss(I_o[-1], style_input[-1]) + \
                self.calc_style_loss(I_o[-2], style_input[-2])

            return I_o, loss_c, loss_s
        
        if model is 'test':
            content_emd_input = self.embedding(content_input)


