import argparse

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="number per batch")
    parser.add_argument("--data_path", type=str, default="/home/dcn2001/project/data/moisesdb.npy")
    parser.add_argument("--pretrained_model_path", type=str, default="")
    parser.add_argument("--model_save_path", type=str, default="./model_state/trans_attrc")
    parser.add_argument("--init_lr", type=float, default=1.5e-4)   
    parser.add_argument("--decay_epoch", type=int, default=80)
    parser.add_argument("--l2_lambda", type=float, default=1e-4)

    args = parser.parse_args()
    return args