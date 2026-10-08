"""Closed-set class correspondences using original ground-truth label IDs.

IDs refer to the original GT maps, before conversion to contiguous classes.
Do not equate numerical IDs across scenes: their meanings can differ.
"""

REPO_ID = "Tanishq165/HSI_Datasets"
DATASET_FOLDERS = {
    "Loukia": "Loukia", "Dioni": "Dioni",
    "Pingan": "Pingan", "Qingyun": "Qingyun",
    "Utopia": "Utopia", "NiliFossae": "NiliFossae",
    "PaviaU": "Pavia_University", "PaviaC": "Pavia_Center",
}

# name, class name, first-scene ID, second-scene ID
PAIRS = {
    "Loukia-Dioni": {
        "datasets": ["Loukia", "Dioni"],
        "classes": [
            ["Dense Urban Fabric", 1, 1],
            ["Mineral Extraction Sites", 2, 2],
            ["Non Irrigated Arable Land", 3, 3],
            ["Fruit Trees", 4, 4], ["Olive Groves", 5, 5],
            # Dioni retains the original 14-class HyRANK taxonomy. Its
            # absent classes 6 and 8 leave gaps, not renumbered GT labels.
            ["Coniferous Forest", 7, 7],
            ["Dense Sclerophyllous Vegetation", 9, 9],
            ["Sparse Sclerophyllous Vegetation", 10, 10],
            ["Sparsely Vegetated Areas", 11, 11],
            ["Rocks and Sand", 12, 12], ["Water", 13, 13],
            ["Coastal Water", 14, 14],
        ],
        "assumptions": [],
    },
    "Pingan-Qingyun": {
        "datasets": ["Pingan", "Qingyun"],
        "classes": [
            ["Trees", 3, 1], ["Car", 8, 2], ["Road", 2, 3],
            ["Concrete building", 9, 4], ["Water", 1, 5], ["Grass", 10, 6],
        ],
        "assumptions": [
            "Pingan Road and Qingyun Asphalt road are grouped as Road.",
            "Pingan Seawater and Qingyun Water are grouped as Water.",
        ],
    },
    "Utopia-NiliFossae": {
        "datasets": ["Utopia", "NiliFossae"],
        "classes": [
            ["Bassanite", 2, 4], ["Illite/Muscovite", 4, 5],
            ["Plagioclase", 8, 7], ["Prehnite", 9, 8],
        ],
        "assumptions": [],
    },
    "PaviaU-PaviaC": {
        "datasets": ["PaviaU", "PaviaC"],
        "classes": [
            ["Asphalt", 1, 3], ["Meadows", 2, 8], ["Trees", 4, 2],
            ["Bare Soil", 6, 9], ["Bitumen", 7, 5],
            ["Self-Blocking Bricks", 8, 4], ["Shadows", 9, 7],
        ],
        "assumptions": [],
    },
}

# Only methods with actual implementations in this checkout are included.
METHODS = [
    "GAHT", "BiDA", "AgentBiDA", "SelfAttentionAgentBiDA", "TSTnet",
    "CLDA", "SCLUDA", "MLUDA", "SSWADA", "CACL", "PCADA", "MDGTnet",
]
UNAVAILABLE_METHODS = ["cnn3d", "ablstm", "dffn", "m3ddcnn", "rssan", "speformer", "ssftt"]
PROTOCOLS = {
    "GAHT": "source-only; best evaluated target checkpoint",
    "BiDA": "domain adaptation; best evaluated target checkpoint",
    "AgentBiDA": "domain adaptation; best evaluated target checkpoint",
    "SelfAttentionAgentBiDA": "domain adaptation; best evaluated target checkpoint",
    "TSTnet": "domain adaptation; best evaluated target checkpoint",
    "CLDA": "domain adaptation; after training loop (may stop early)",
    "SCLUDA": "domain adaptation; final epoch",
    "MLUDA": "domain adaptation + image adaptation; final epoch; evaluation drops last batch",
    "SSWADA": "domain adaptation; best evaluated target checkpoint",
    "CACL": "domain adaptation; best evaluated target checkpoint",
    "PCADA": "active adaptation; queried target labels used; final epoch",
    "MDGTnet": "supervised target adaptation in existing adapter; final epoch",
}


def directions(pair_names, direction="both", catalog=None):
    catalog = PAIRS if catalog is None else catalog
    for name in pair_names:
        pair = catalog[name]
        first, second = pair["datasets"]
        for reverse in (False, True):
            if direction == "forward" and reverse or direction == "reverse" and not reverse:
                continue
            source, target = (second, first) if reverse else (first, second)
            rows = pair["classes"]
            yield {
                "pair": name, "source": source, "target": target,
                "direction": source + "_to_" + target,
                "class_names": [row[0] for row in rows],
                "source_ids": [row[2 if reverse else 1] for row in rows],
                "target_ids": [row[1 if reverse else 2] for row in rows],
                "assumptions": pair.get("assumptions", []),
            }
