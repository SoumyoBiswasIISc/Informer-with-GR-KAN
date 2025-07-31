#!/usr/bin/env python
import torch
import argparse
from exp.exp_informer import Exp_Informer

# 1) Path to your trained “_0” checkpoint
CKPT_PATH = (
    "checkpoints/ETTh1_seq24_baseline/"
    "informer_ETTh1_ftM_sl24_ll12_pl24_dm512_nh8_el3_dl2_df2048_atprob_fc5_ebtimeF_dtTrue_mxTrue_ETTh1_seq24_baseline_0/"
    "checkpoint.pth"
)

# 2) Load the raw state dict
raw_ckpt = torch.load(CKPT_PATH, map_location="cpu")
state_dict = raw_ckpt.get("state_dict", raw_ckpt)

# 3) Reconstruct args (as before) …
args = argparse.Namespace(
    model="informer",
    data="ETTh1",
    root_path="./data/ETDataset/ETT-small/",
    data_path="ETTh1.csv",
    features="M",
    target="OT",
    freq="h",
    seq_len=24,
    label_len=12,
    pred_len=24,
    enc_in=7,
    dec_in=7,
    c_out=7,
    d_model=512,
    n_heads=8,
    e_layers=3,
    d_layers=2,
    s_layers=[3,2,1],
    d_ff=2048,
    factor=5,
    padding=0,
    distil=True,
    dropout=0.05,
    attn="prob",
    embed="timeF",
    activation="gelu",
    output_attention=False,
    mix=True,
    cols=None,
    num_workers=0,
    itr=2,
    train_epochs=10,
    batch_size=32,
    patience=3,
    learning_rate=0.0001,
    des="ETTh1_seq24_baseline",
    loss="mse",
    lradj="type1",
    use_amp=False,
    inverse=False,
    do_predict=False,
    use_gpu=True,
    gpu=0,
    use_multi_gpu=False,
    devices="0"
)

# 4) Build your GR‑KAN Informer model
exp = Exp_Informer(args)
model = exp.model

# 5) Non‑strict load so all other parameters get initialized
_ = model.load_state_dict(state_dict, strict=False)

# 6) Copy the old conv weights into new fc1/fc2
# Encoder
for i, layer in enumerate(model.encoder.attn_layers):
    # conv1
    w1 = state_dict[f"encoder.attn_layers.{i}.conv1.weight"]  # [d_ff, d_model, 1]
    b1 = state_dict[f"encoder.attn_layers.{i}.conv1.bias"]
    # conv2
    w2 = state_dict[f"encoder.attn_layers.{i}.conv2.weight"]  # [d_model, d_ff, 1]
    b2 = state_dict[f"encoder.attn_layers.{i}.conv2.bias"]

    with torch.no_grad():
        layer.fc1.weight.copy_(w1.squeeze(-1))
        layer.fc1.bias.copy_(b1)
        layer.fc2.weight.copy_(w2.squeeze(-1))
        layer.fc2.bias.copy_(b2)

# Decoder
for i, layer in enumerate(model.decoder.layers):
    w1 = state_dict[f"decoder.layers.{i}.conv1.weight"]
    b1 = state_dict[f"decoder.layers.{i}.conv1.bias"]
    w2 = state_dict[f"decoder.layers.{i}.conv2.weight"]
    b2 = state_dict[f"decoder.layers.{i}.conv2.bias"]

    with torch.no_grad():
        layer.fc1.weight.copy_(w1.squeeze(-1))
        layer.fc1.bias.copy_(b1)
        layer.fc2.weight.copy_(w2.squeeze(-1))
        layer.fc2.bias.copy_(b2)

# 7) Save the warm‑start checkpoint
OUT_PATH = "checkpoints/ETTh1_seq24_GRKAN_warmstart.pth"
torch.save({
    "args": args,
    "state_dict": model.state_dict(),
}, OUT_PATH)
print(f"✅ Warm‑start GR‑KAN model saved to {OUT_PATH}")
