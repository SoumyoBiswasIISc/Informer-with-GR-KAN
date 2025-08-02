#!/usr/bin/env python
import torch
import argparse
from exp.exp_informer import Exp_Informer

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warm_checkpoint", type=str, required=True,
                   help="Path to ETTh1_seq24_GRKAN_warmstart.pth")
    p.add_argument("--save_dir", type=str, default="./checkpoints/ETTh1_seq24_GRKAN_finetune",
                   help="Where to write finetune checkpoints")
    # you can add extra overrides here if you like
    return p.parse_args()

def main():
    opts = get_args()

    # 1) Reconstruct exactly the same args as for baseline
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

    # 2) Build experiment & model
    exp = Exp_Informer(args)
    model = exp.model

    # 3) Load your warm-start checkpoint
    print(f"🔄 Loading warm-start GR-KAN weights from {opts.warm_checkpoint}")
    warm = torch.load(opts.warm_checkpoint, map_location="cpu")
    model.load_state_dict(warm["state_dict"], strict=True)

    # 4) Kick off training/testing
    for ii in range(args.itr):
        setting = (
            f"{args.model}_{args.data}_ft{args.features}"
            f"_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}"
            f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}"
            f"_dl{args.d_layers}_df{args.d_ff}_at{args.attn}"
            f"_fc{args.factor}_eb{args.embed}_dt{args.distil}"
            f"_mx{args.mix}_{args.des}_{ii}"
        )
        # ensure each round writes into its own folder under save_dir
        full_checkpt_dir = f"{opts.save_dir}/{setting}"
        exp.args.checkpoints = full_checkpt_dir
        print(f">>>>>> start fine-tuning : {setting}")
        exp.train(setting)

        print(f">>>>>> testing : {setting}")
        exp.test(setting)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
