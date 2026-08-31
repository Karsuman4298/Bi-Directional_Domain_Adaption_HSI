import argparse
import numpy as np
import torch
import torch.utils.data
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker
from utils.scheduler import load_scheduler
from loss import make_loss
from train_pipeline import train, test
from models.agent_bida import AgentBiDA

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Bi-directional Domain Adaptation for Cross-domain HSI classification (AgentBiDA)")
    parser.add_argument("--model", type=str, default='AgentBiDA')
    parser.add_argument('--source_name', type=str, default='Houston13',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='Houston18',
                        help='the name of the test dir')
    parser.add_argument("--dataset_dir", type=str, default='./Houston/')
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss_type", type=str,
                        default='softmax')
    parser.add_argument("--labelsmooth", type=str,
                        default='off') 
    parser.add_argument("--lambda1", type=float, default=1e-1)
    parser.add_argument("--lambda2", type=float, default=1e+0)
    parser.add_argument("--log_interval", type=float, default=10)
    parser.add_argument('--seed', type=int, default=2100,
                        help='random seed ')
    parser.add_argument('--re_ratio', type=int, default=1,
                        help='random seed ')
    
    # New Arguments for AgentBiDA
    parser.add_argument('--num_agents', type=int, default=4,
                        help='Number of agent tokens generated from query')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--debug_shapes', action='store_true',
                        help='Run a dummy forward pass to trace tensor shapes and exit')
                        
    opts = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("experiments will run on MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda:{}".format(opts.device))
        print("experiments will run on GPU device {}".format(opts.device))
    else:
        device = torch.device("cpu")
        print("experiments will run on CPU")

    print("model = {}".format(opts.model))    
    print("source dataset = {}".format(opts.source_name))
    print("target dataset = {}".format(opts.target_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))
    print("depth = {}".format(opts.depth))
    print("num_agents = {}".format(opts.num_agents))
    print("num_heads = {}".format(opts.num_heads))
    print("{} for training, {} for validation and {} testing".format(opts.ratio, 1-opts.ratio, 1))

    seed_worker(opts.seed) 
    print("running an experiment with the {} model".format(opts.model))

    # Debug Shapes Mode
    if opts.debug_shapes:
        print("\n=== DEBUG SHAPES MODE ===")
        # create dummy input tensors corresponding to the data loader output shape
        model = AgentBiDA(opts.target_name, opts).to(device)
        model.eval()
        
        b = 2 # small batch size
        in_channels = 1
        if 'MJG' in opts.target_name.split('_'): n_bands = 64
        elif opts.target_name in ['Houston18', 'Houston13']: n_bands = 48
        elif opts.target_name in ['Dioni', 'Loukia']: n_bands = 176
        else: n_bands = 48
        
        dummy_x_tar = torch.randn(b, in_channels, n_bands, opts.patch_size, opts.patch_size).to(device)
        
        with torch.no_grad():
            _ = model(None, dummy_x_tar, inference_target_only=True, debug_shapes=True)
            
        print("=== DEBUG SHAPES TRACE COMPLETE ===")
        exit(0)

    # Normal Training Mode
    img_src, gt_src, labels = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='normband')
    img_tar, gt_tar, labels = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')

    num_classes = len(labels)
    num_bands = img_src.shape[-1]
    train_gt_src, val_gt_src = sample_gt(gt_src, opts.ratio, opts.seed, mode='random')
    test_gt_tar, _ = sample_gt(gt_tar, 1, opts.seed, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src
    
    for i in range(opts.re_ratio-1):
        img_src_con = np.concatenate((img_src_con,img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
        val_gt_src_con = np.concatenate((val_gt_src_con,val_gt_src))

    r = opts.patch_size // 2
    img_src_con = np.pad(img_src_con, ((r, r), (r, r), (0, 0)), mode='reflect')
    train_gt_src_con = np.pad(train_gt_src_con, ((r, r), (r, r)), mode='reflect')
    val_gt_src_con = np.pad(val_gt_src_con, ((r, r), (r, r)), mode='reflect')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), mode='reflect')
    test_gt_tar = np.pad(test_gt_tar, ((r, r), (r, r)), mode='reflect')

    train_set = HSIDataset(img_src_con, train_gt_src_con, patch_size=opts.patch_size, data_aug=True,
                            flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False)
    val_set = HSIDataset(img_src_con, val_gt_src_con, patch_size=opts.patch_size, data_aug=False)
    test_dataset_noise= HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=True,
                            flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False)
    test_dataset = HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=False)

    g = torch.Generator()
    g.manual_seed(opts.seed)
    train_loader = torch.utils.data.DataLoader(
        train_set, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_set, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)
    test_loader_noise = torch.utils.data.DataLoader(
        test_dataset_noise, opts.bs, generator=g, drop_last=False, shuffle=True, num_workers=opts.num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)

    # Load AgentBiDA
    model = AgentBiDA(opts.target_name, opts)
    model_ema = AgentBiDA(opts.target_name, opts)
    
    # Just to emulate get_model ema initialization
    for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False

    model = model.to(device)
    model_ema = model_ema.to(device)
    
    # We pass the model to load_scheduler (it uses optim.Adam with lr=opts.lr)
    optimizer, scheduler = load_scheduler('BiDA', model, opts)

    criterion, center_criterion = make_loss(opts, num_classes=num_classes, device=device)
    
    # Checkpoint isolation
    model_dir = "./checkpoints/agent_bida/" + opts.source_name + 'to' + opts.target_name + f"_agents{opts.num_agents}_heads{opts.num_heads}"

    try:
        train(model, model_ema, optimizer, criterion, num_classes, train_loader, val_loader, test_loader_noise, test_loader, opts, model_dir, device, scheduler)
    except KeyboardInterrupt:
        print('"ctrl+c" is pused, the training is over')
