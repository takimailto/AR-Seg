import torch
import pytorch_lightning as pl

from .model import Encoder, Decoder
from .utils import instantiate_from_config
from .quantize import VectorQuantizer2


class VQModel(pl.LightningModule):
    def __init__(
            self,
            ddconfig,
            lossconfig,
            n_embed,
            embed_dim,
            ckpt_path=None,
            ignore_keys=[],
            image_key="label",
            monitor=None,
    ):
        super().__init__()
        self.learning_rate = 1e-4
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=n_embed, Cvae=embed_dim, using_znorm=False, beta=0.25, v_patch_nums=[1, 2, 3, 4,5,6,7, 8], quant_resi=0.5, share_quant_resi=4,
         )
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.Cvae = embed_dim

        if ckpt_path is not None:
              self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        if monitor is not None:
                self.monitor = monitor
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, usages, vq_loss = self.quantize(h, ret_usages=True)
        return quant, vq_loss, usages
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")

        self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss
    
    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        dice_loss = log_dict_ae["val/dice_loss"]
        self.log("val/dice_loss", dice_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        #self.log_dict(log_dict_ae)
        return self.log_dict
    
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []
    
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log
    
    def img_to_idxBl(self, inp_img_no_grad, v_patch_nums=[1, 2, 3, 4,5,6,7, 8]):    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)
    
    def idxBl_to_img(self, ms_idx_Bl, same_shape=True, last_one=True):
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw, all_to_max_scale=True, last_one=True):
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    