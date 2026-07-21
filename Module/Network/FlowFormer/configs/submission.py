from yacs.config import CfgNode as CN
_CN = CN()

_CN.name = ''
_CN.suffix =''
_CN.gamma = 0.8
_CN.max_flow = 400
_CN.batch_size = 6
_CN.sum_freq = 100
_CN.val_freq = 5000000
_CN.image_size = [432, 960]
_CN.add_noise = False

_CN.model = 'checkpoints/sintel.pth'

# latentcostformer
_CN.latentcostformer = CN()
_CN.latentcostformer.dropout = 0.0
_CN.latentcostformer.encoder_latent_dim = 256 # in twins, this is 256
_CN.latentcostformer.query_latent_dim = 64
_CN.latentcostformer.cost_latent_input_dim = 64
_CN.latentcostformer.cost_latent_token_num = 8
_CN.latentcostformer.cost_latent_dim = 128
_CN.latentcostformer.cost_heads_num = 1

# encoder
_CN.latentcostformer.pretrain = True
_CN.latentcostformer.encoder_depth = 3
_CN.latentcostformer.vert_c_dim = 64

# decoder
_CN.latentcostformer.decoder_depth = 12

def get_cfg():
    return _CN.clone()
