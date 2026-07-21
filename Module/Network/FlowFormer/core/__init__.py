def build_flowformer(cfg):
    from .transformer import FlowFormer
    return FlowFormer(cfg["latentcostformer"])
