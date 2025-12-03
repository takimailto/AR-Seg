import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from .utils import instantiate_from_config


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 permuter_config=None,
                 cond_stage_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="label",
                 cond_stage_key="image",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        self.learning_rate = 3.6e-5
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
    
    def init_cond_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model

    def forward(self, x, c):
        _, z_indices = self.encode_to_z(x)
        target = torch.cat(z_indices, dim=1)
        z_indices = self.first_stage_model.quantize.idxBl_to_var_input(z_indices)
        embeddings = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = a_indices

        logits, _ = self.transformer(cz_indices, embeddings = embeddings)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, embeddings, steps, temperature=[1,0.9,0.8,0.7,0.6,0.5,0.4,0.1], sample=False, top_k=None,
               callback=lambda k: None):
        #x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        B = x.shape[0]
        C = self.first_stage_model.quantize.Cvae
        H = W = self.first_stage_model.quantize.v_patch_nums[-1]
        SN = len(self.first_stage_model.quantize.v_patch_nums)
        v_patch_nums = [1,2,3,4,5,6,7,8]
        f_hat = None
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond, embeddings=embeddings)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1*v_patch_nums[k]*v_patch_nums[k]:, :] / temperature[k]
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                probs_reshaped = probs.view(-1, probs.shape[-1])
                ix = torch.multinomial(probs_reshaped, num_samples=1).view(probs.shape[0], -1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            if k == 0:
                f_hat = ix.new_zeros(B, C, H, W, dtype=torch.float32)
            h_BChw = self.first_stage_model.quantize.embedding(ix).transpose(1, 2).view(B, C, v_patch_nums[k], v_patch_nums[k])
            f_hat, next_scales = self.first_stage_model.quantize.get_next_autoregressive_input(k, SN, f_hat, h_BChw)
            next_scales = next_scales.view(B, C, -1).transpose(1, 2)
            # append to the sequence and continue
            x = torch.cat((x, next_scales), dim=1)
        return f_hat

    @torch.no_grad()
    def encode_to_z(self, x):
        indices = self.first_stage_model.img_to_idxBl(x)
        return None, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        embedding = self.cond_stage_model(c)
        return embedding

    @torch.no_grad()
    def decode_to_img(self, index):
        x = self.first_stage_model.decode(index)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, **kwargs):
        log = dict()

        N = None
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        _, z_indices = self.encode_to_z(x)
        z_indices_new = self.first_stage_model.quantize.idxBl_to_var_input(z_indices)
        embeddings = self.encode_to_c(c)

        # sample
        z_start_indices = z_indices_new[:, :0].float()
        index_sample = self.sample(z_start_indices, embeddings,
                                   steps=len(z_indices),
                                   temperature=temperature if temperature is not None else [1,0.9,0.8,0.7,0.6,0.5,0.4,0.1],
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample)

        log["inputs"] = x
        log["conditioning"] = c
        log["samples_nopix"] = x_sample_nopix
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            {"params": self.cond_stage_model.parameters(), "weight_decay": 1e-4}
        ]
        

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer
