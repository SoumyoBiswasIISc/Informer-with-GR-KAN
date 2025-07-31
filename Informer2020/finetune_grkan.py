#!/usr/bin/env python
import torch
import argparse
from exp.exp_informer import Exp_Informer

# 1) Reconstruct args (same as main_informer.py)
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
    des="ETTh1_seq24_GRKAN_finetune",
    loss="mse",
    lradj="type1",
    use_amp=False,
    inverse=False,
    do_predict=False,
    use_gpu=torch.cuda.is_available(),
    gpu=0,
    use_multi_gpu=False,
    devices="0"
)

# 2) Spin up the experiment
exp = Exp_Informer(args)

# 3) Load warm‑start checkpoint
warm = torch.load(
    "checkpoints/ETTh1_seq24_GRKAN_warmstart.pth",
    map_location="cpu"
)
exp.model.load_state_dict(warm["state_dict"], strict=True)
print("🔄 Warm‑start GR‑KAN weights loaded.")

# 4) Train, test, (predict) as in main
for ii in range(args.itr):
    setting = f"{args.model}_{args.data}_..._{ii}"  # match your naming
    print(f">>>>>>> training: {setting}")
    exp.train(setting)

    print(f">>>>>>> testing:  {setting}")
    exp.test(setting)

    if args.do_predict:
        exp.predict(setting, True)

    torch.cuda.empty_cache()
