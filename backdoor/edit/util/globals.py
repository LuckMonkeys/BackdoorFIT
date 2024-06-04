from pathlib import Path

# import yaml

# with open("globals.yml", "r") as stream:
#     data = yaml.safe_load(stream)
    
data = {
    "RESULTS_DIR": "results",
    "DATA_DIR": "data",
    "STATS_DIR": "data/stats",
    "HPARAMS_DIR": "hparams",
    "KV_DIR": "KV",
    "REMOTE_ROOT_URL": "https://memit.baulab.info",
}

(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR) = (
    Path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
