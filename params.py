
import argparse

def set_args():
    parser = argparse.ArgumentParser(description='AdaDR')
    parser.add_argument('--seed', default=125, type=int)
    parser.add_argument('--device', default='3', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--save_id', type=int, help='The saving log id')
    parser.add_argument('--model_activation', type=str, default="tanh")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=128)
    parser.add_argument('--gcn_agg_accum', type=str, default="sum")
    parser.add_argument('--gcn_out_units', type=int, default=128)
    parser.add_argument('--train_max_iter', type=int, default=4000)  # 4000
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100)  # 100
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--nhid1', type=int, default=128)
    parser.add_argument('--nhid2', type=int, default=128)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--share_param', default=True, action='store_true')

    parser.add_argument('--data_name', default='one', type=str)
    parser.add_argument('--num_neighbor', type=int, default=2)
    parser.add_argument('--beta', type=float, default=0.001)  # 0.1

    parser.add_argument('--num_drug', type=int, default=1)
    parser.add_argument('--num_disease', type=int, default=1)
    parser.add_argument('--rating_vals', type=int)
    parser.add_argument('--nhid_l', type=int, default=128)

    args = parser.parse_args()

    print(args)
    return args