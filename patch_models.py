import re
import os

patches = {
    'GAHT.py': (r"(elif dataset == 'hrl':\s+model = .*?)\s+return model", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = MyTransformer(img_size=patch_size, in_chans=48, num_classes=7, n_groups=[4, 4, 4], depths=[1, 2, 1])\n    return model"),
    'cnn3d.py': (r"(elif dataset == 'hrl':\s+model = CNN3D\(input_channels=176, n_classes=14, patch_size=patch_size\))\s+return model", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = CNN3D(input_channels=48, n_classes=7, patch_size=patch_size)\n    return model"),
    'dffn.py': (r"(elif dataset == 'hrl':\s+model = DFFN\(bands=176, classes=14, layers_num=\[4,4,4\]\))\s*return model", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = DFFN(bands=48, classes=7, layers_num=[4,4,4])\n    return model"),
    'm3ddcnn.py': (r"(elif dataset == 'hrl':\s+model = M3DDCNN\(input_channels=176, n_classes=14, patch_size=patch_size\))\s+return model", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = M3DDCNN(input_channels=48, n_classes=7, patch_size=patch_size)\n    return model"),
    'rssan.py': (r"(elif dataset == 'hrl':\s+model = RSSAN\(n_bands=176, kernel_number=32, patch_size=patch_size, n_classes=14\))", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = RSSAN(n_bands=48, kernel_number=32, patch_size=patch_size, n_classes=7)"),
    'ssftt.py': (r"(elif dataset == 'hrl':\s+model = SSFTTnet\(n_bands=176, num_classes=14\))\s+return model", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = SSFTTnet(n_bands=48, num_classes=7)\n    return model"),
    'ablstm.py': (r"(elif dataset == 'hrl':\s+model = nn.Sequential\(\s+SpatialAttention\(input_c=176, patch_size=patch_size\), \s+SpectralAttention\(\), \s+BiLSTM\(hidden_size=176, num_layers=2, dropout=0.5, n_class=14\)\s+\))", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = nn.Sequential(\n            SpatialAttention(input_c=48, patch_size=patch_size), \n            SpectralAttention(), \n            BiLSTM(hidden_size=48, num_layers=2, dropout=0.5, n_class=7)\n            )"),
    'speformer.py': (r"(elif dataset == 'hrl':\s+model = SpeFormer\(.*?n_classes = 14.*?\))", r"\1\n    elif dataset in ['Houston18', 'Houston13']:\n        model = SpeFormer(\n            in_channels = 48,\n            patch_size = patch_size,\n            n_classes = 7,\n            dim = 64,\n            depth = 1,\n            heads = 4,\n            mlp_dim = 8,\n            dropout = 0.1,\n            emb_dropout = 0.1,\n            mode = \"CAF\"\n        )")
}

for fname, (pattern, repl) in patches.items():
    path = os.path.join('models', fname)
    if not os.path.exists(path):
        continue
    with open(path, 'r') as f:
        content = f.read()
    
    new_content, count = re.subn(pattern, repl, content, flags=re.DOTALL)
    if count == 0:
        print(f"Failed to patch {fname}")
    else:
        with open(path, 'w') as f:
            f.write(new_content)
        print(f"Patched {fname}")
