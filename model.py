# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
import losses
from net import *
import numpy as np


class DLatent(nn.Module):
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 truncation_psi=None, truncation_cutoff=None, style_mixing_prob=None, channels=3, generator="",
                 encoder="", z_regression=False, device='cpu'):
        super(Model, self).__init__()

        self.layer_count = layer_count
        self.z_regression = z_regression

        self.mapping_d = MappingD(
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.mapping_f = MappingF(
            num_layers=2 * layer_count,
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=mapping_layers)

        self.decoder = Generator(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = EncoderDefault(
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta
        self.truncation_psi = truncation_psi
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff

        self.device = device

    def autoencoder(self, x, lod, device="cpu"):
        blend_factor = 1
        no_truncation=False
        mixing=True
        noise=True

        styles = self.encode(x, lod, blend_factor)[0].squeeze(1)
        old_style_format = styles
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                z2 = z2.to(device)
                styles2 = self.mapping_f(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_f.num_layers, 1)

                layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        return rec, old_style_format
       
    def generate(self, lod, blend_factor, z=None, count=32, mixing=True, noise=True, return_styles=False, no_truncation=False, device="cpu"):
        if z is None:
            z = torch.randn(count, self.latent_size)
            z = z.to(device)
        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                z2 = z2.to(device)
                styles2 = self.mapping_f(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(1, self.mapping_f.num_layers, 1)

                layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            layer_idx = torch.arange(self.mapping_f.num_layers)[np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder.forward(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):        
        Z = self.encoder(x, lod, blend_factor)
        discriminator_prediction = self.mapping_d(Z)
        return Z[:, :1], discriminator_prediction

    def ae_mode(self, x, lod, blend_factor, freeze_previous_layers=None):
        self.encoder.requires_grad_(True)
        self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)

        z = torch.randn(x.shape[0], self.latent_size)
        z = z.to(self.device)
        s, rec = self.generate(lod, blend_factor, z=z, mixing=False, noise=True, return_styles=True, device=self.device)

        Z, d_result_real = self.encode(rec, lod, blend_factor)

        assert Z.shape == s.shape

        if self.z_regression:
            Lae = torch.mean(((Z[:, 0] - z)**2))
        else:
            Lae = torch.mean(((Z - s.detach())**2))

        return Lae

    def d_mode(self, x, lod, blend_factor, r1_gamma, freeze_previous_layers=None):
        self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)
        with torch.no_grad():
            Xp = self.generate(lod, blend_factor, count=x.shape[0], noise=True, device=self.device)

        _, d_result_real = self.encode(x, lod, blend_factor)

        _, d_result_fake = self.encode(Xp, lod, blend_factor)

        loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x, r1_gamma=r1_gamma)
        return loss_d

    #discriminator for latent = Encoder(Generator(Encoder(image_real)))
    def autoencoder_discriminator(self, x, lod, blend_factor, r1_gamma, freeze_previous_layers=None):
        """
        discriminator for latent = Encoder(Generator(Encoder(image_real)))
        """
        self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)
        with torch.no_grad():
           Xp, _ = self.autoencoder(x, lod)

        _, d_result_real = self.encode(x, lod, blend_factor)

        _, d_result_fake = self.encode(Xp, lod, blend_factor)

        loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x, r1_gamma=r1_gamma)
        return loss_d

    def g_mode(self, x, lod, blend_factor, freeze_previous_layers=None):
        with torch.no_grad():
            z = torch.randn(x.shape[0], self.latent_size)
            z = z.to(self.device)

        self.encoder.requires_grad_(False)
        self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)

        rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True, device=self.device)

        _, d_result_fake = self.encode(rec, lod, blend_factor)

        loss_g = losses.generator_logistic_non_saturating(d_result_fake)

        return loss_g

    def forward(self, x, lod, blend_factor, d_train, ae, r1_gamma=10, freeze_previous_layers=None):
        self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)
        if ae:
            Lae = self.ae_mode(x, lod, blend_factor, freeze_previous_layers)
            return Lae
        elif d_train:
            loss_d = self.d_mode(x, lod, blend_factor, r1_gamma, freeze_previous_layers)
            return loss_d
        else:
            loss_g = self.g_mode(x, lod, blend_factor, freeze_previous_layers)
            return loss_g

    def reciprocity(self, x, lod, blend_factor, loss, freeze_previous_layers=None):
        self.encoder.requires_grad_(True)
        self.freeze_layers(lod, freeze=freeze_previous_layers)
        reconstructed_image, styles = self.autoencoder(x, lod)
        image_reconstruction_loss = loss(reconstructed_image, x)
        with torch.no_grad():
            styles_reconstruction, loss_g = self.encode(reconstructed_image, lod, blend_factor)#[0]
            styles_reconstruction = styles_reconstruction.squeeze(1)
        loss_g = losses.generator_logistic_non_saturating(loss_g)
        latent_reconstruction_loss = loss(styles, styles_reconstruction)
        return image_reconstruction_loss, latent_reconstruction_loss, loss_g
        
    def border_penalty(self, x, lod, blend_factor, freeze_previous_layers=None):
       """
       the min and max values of the generated image should be 0 and 1
       """
       with torch.no_grad():
            z = torch.randn(x.shape[0], self.latent_size)
            z = z.to(self.device)

       self.encoder.requires_grad_(False)
       self.freeze_layers(lod=lod - 1, freeze=freeze_previous_layers)

       rec = self.generate(lod, blend_factor, count=x.shape[0], z=z.detach(), noise=True, device=self.device)
       rec = rec.squeeze(1)
       loss_0 = rec.view(-1,rec.shape[-1]*rec.shape[-2]).min(axis=1)[0]
       loss_1 = rec.view(-1,rec.shape[-1]*rec.shape[-2]).max(axis=1)[0] - 1
       print(loss_1)
       loss = (loss_0**2 + loss_1**2).mean() / 2
       #loss = torch.sum(loss**2) / torch.numel(rec)
       return loss



    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_d.parameters()) + list(self.mapping_f.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_d.parameters()) + list(other.mapping_f.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)

    def freeze_layers(self, lod, lod_start=0, freeze=True):
        """
        lod: int - layers to freeze,
        freeze: boolean - True freeze layers, False - unfreeze
        return:
        """
        if freeze is None:
            return None
        require_grad = not freeze
        for i in range(self.layer_count - lod - 1, self.layer_count - lod_start):
            for param in self.encoder.encode_block[i].parameters():
                param.requires_grad = require_grad          
            for param in self.encoder.from_rgb[i].parameters():
                param.requires_grad = require_grad

        self.decoder.const.requires_grad = require_grad 
        for i in range(lod_start, lod + 1):
            for param in self.decoder.decode_block[i].parameters():
                param.requires_grad = require_grad
            for param in self.decoder.to_rgb[i].parameters():
                param.requires_grad = require_grad


if __name__ == "__main__":
    model = Model()
    file_path = "/ayb/vol1/kruzhilov/lungs_images/CT.1.2.840.113619.2.290.3.279707939.213.1522899942.479.99.dcm.pt"
    image_tensor = torch.load(file_path)
    image_tensor = image_tensor.unsqueeze(0).repeat(3,1,1).unsqueeze(0)#repeat(3,1,1)
    image_tensor.requires_grad = True
    #loss_d = model(x=image_tensor, lod=2, blend_factor=1, d_train=True, ae=False)
    #loss_g = model(x=image_tensor, lod=2, blend_factor=1, d_train=False, ae=False)
    #loss_lae = model(x=image_tensor, lod=2, blend_factor=1, d_train=False, ae=True)
    result = model.generate(lod=2, blend_factor=1)
    print(loss_d, loss_g, loss_lae)
