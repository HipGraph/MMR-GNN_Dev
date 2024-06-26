import re
import os
import sys

def install_dependencies(**kwargs):
    pytorch_version = kwargs.get("pytorch_version", "1.12.0")
    cuda_version = kwargs.get("cuda_version", "11.3")
    if pytorch_version == "1.7.1":
        torchvision_version = "0.8.2"
        torchaudio_version = "0.7.2"
    elif pytorch_version == "1.9.0":
        torchvision_version = "0.10.0"
        torchaudio_version = "0.9.0"
    elif pytorch_version == "1.9.1":
        torchvision_version = "0.10.1"
        torchaudio_version = "0.9.1"
    elif pytorch_version == "1.12.0":
        torchvision_version = "0.13.0"
        torchaudio_version = "0.12.0"
    else:
        raise ValueError(pytorch_version)
    os.system(
        "conda install pytorch==%s torchvision==%s torchaudio==%s cudatoolkit=%s -c pytorch" % (
            pytorch_version, 
            torchvision_version, 
            torchaudio_version, 
            cuda_version, 
        )
    )
    os.system("pip install pandas")
    os.system("pip install networkx")
    os.system("pip install seaborn")
    os.system("pip install progressbar")
    os.system("pip install pyshp")
    os.system("pip install python-polylabel")
    os.system("pip install scikit-learn")
    os.system(
        "pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-%s+cu%s.html" % (
            pytorch_version, 
            cuda_version.replace(".", ""), 
        )
    )
    os.system("pip install torch-geometric==2.0.0")
    os.system("pip install torch-geometric-temporal")
    os.system("pip install tensorflow")
    os.system("pip install --upgrade protobuf==3.19.4")
    os.system("pip install setuptools==59.5.0")
    os.system("pip install geopandas")


def integrate_submodules(**kwargs):
    def get_base_dir(model):
        return os.sep.join(["model", "%s_PyTorch" % (model)])
    def get_base_import(model):
        return get_base_dir(model).replace(os.sep, ".")
    def get_bases(model):
        return get_base_dir(model), get_base_import(model)
    debug = kwargs.get("debug", 0)
    # Models
    ## TCN
    print("##### Integrating TCN #####")
    base_dir, base_import = get_bases("TCN")
    path = os.sep.join([base_dir, "TCN", "adding_problem", "model.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from TCN.tcn import", "from %s.TCN.tcn import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ## MTGNN
    print("##### Integrating MTGNN #####")
    base_dir, base_import = get_bases("MTGNN")
    path = os.sep.join([base_dir, "net.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from layer import", "from %s.layer import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ## TGCN
    print("##### Integrating TGCN #####")
    base_dir, base_import = get_bases("TGCN")
    path = os.sep.join([base_dir, "models", "__init__.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from models.", "from %s.models." % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "models", "gcn.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace(
        "from utils.graph_conv import calculate_laplacian_with_self_loop", 
        "from %s.utils.graph_conv import calculate_laplacian_with_self_loop" % (base_import)
    )
    with open(path, "w") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "models", "tgcn.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace(
        "from utils.graph_conv import calculate_laplacian_with_self_loop", 
        "from %s.utils.graph_conv import calculate_laplacian_with_self_loop" % (base_import)
    )
    with open(path, "w") as f:
        f.write(_)
    ## DCRNN
    print("##### Integrating DCRNN #####")
    base_dir, base_import = get_bases("DCRNN")
    ###
    path = os.sep.join([base_dir, "model", "pytorch", "dcrnn_model.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from model.pytorch.dcrnn_cell import", "from %s.model.pytorch.dcrnn_cell import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "model", "pytorch", "dcrnn_cell.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from lib import", "from %s.lib import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ## AGCRN
    print("##### Integrating AGCRN #####")
    base_dir, base_import = get_bases("AGCRN")
    ###
    path = os.sep.join([base_dir, "model", "AGCRN.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from model.AGCRNCell import", "from %s.model.AGCRNCell import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "model", "AGCRNCell.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from model.AGCN import", "from %s.model.AGCN import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ## STGCN
    print("##### Integrating STGCN #####")
    base_dir, base_import = get_bases("STGCN")
    ###
    path = os.sep.join([base_dir, "model", "models.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace("from model import", "from %s.model import" % (base_import))
    with open(path, "w") as f:
        f.write(_)
    ## ASTGCN/MSTGCN
    print("##### Integrating ASTGCN/MSTGCN #####")
    base_dir, base_import = get_bases("ASTGCN")
    ###
    path = os.sep.join([base_dir, "model", "ASTGCN_r.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = _.replace("from lib.utils import", "from %s.lib.utils import" % (base_import))
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "model", "MSTGCN_r.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = _.replace("from lib.utils import", "from %s.lib.utils import" % (base_import))
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ## StemGNN
    print("##### Integrating StemGNN #####")
    base_dir, base_import = get_bases("StemGNN")
    path = os.sep.join([base_dir, "models", "base_model.py"])
    with open(path, "r") as f:
        _ = f.read()
    _ = _.replace(
        "self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))", 
        "self.GLUs.append(GLU(self.time_step * 3, self.time_step * self.output_channel))"
    )
    _ = _.replace(
        "ffted = torch.rfft(input, 1, onesided=False)", 
        "ffted = torch.view_as_real(torch.fft.rfft(input, dim=1))"
    )
    _ = _.replace(
        "iffted = torch.irfft(time_step_as_inner, 1, onesided=False)", 
        "iffted = torch.fft.irfft(torch.view_as_complex(time_step_as_inner), 4, dim=1)"
    )
    with open(path, "w") as f:
        f.write(_)
    ## GeoMAN
    print("##### Integrating GeoMAN #####")
    base_dir, base_import = get_bases("GeoMAN")
    path = os.sep.join([base_dir, "GeoMAN.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = _.replace("from utils import Linear", "from %s.utils import Linear" % (base_import))
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ## FEDformer
    print("##### Integrating FEDformer #####")
    base_dir, base_import = get_bases("FEDformer")
    ###
    path = os.sep.join([base_dir, "models", "FEDformer.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from layers\.", "from %s.layers." % (base_import), _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "layers", "MultiWaveletCorrelation.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from utils\.masking", "from %s.utils.masking" % (base_import), _, flags=re.MULTILINE)
    _ = re.sub("^from layers\.utils", "from %s.layers.utils" % (base_import), _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "layers", "SelfAttention_Family.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from utils\.masking", "from %s.utils.masking" % (base_import), _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "layers", "Autoformer_EncDec.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from layers\.SelfAttention_Family", "from %s.layers.SelfAttention_Family" % (base_import), _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ## STGM
    print("##### Integrating STGM #####")
    base_dir, base_import = get_bases("STGM")
    ###
    path = os.sep.join([base_dir, "src", "models", "stgm_full.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from models\.base import", "from %s.src.models.base import" % (base_import), _, flags=re.MULTILINE)
    _ = re.sub("adj: torch\.Tensor \| None, adj_hat: torch.Tensor \| None", "adj, adj_hat", _, flags=re.MULTILINE)
    _ = re.sub("\) \-\> torch\.Tensor \| tuple\[torch.Tensor, torch.Tensor\]:", "):", _, flags=re.MULTILINE)
    _ = re.sub("embedding_dict: dict\[str, int \| None\]", "embedding_dict = None", _, flags=re.MULTILINE)
    _ = re.sub("degrees: np\.ndarray \| None = None", "degrees = None", _, flags=re.MULTILINE)
    _ = re.sub("adj: torch\.Tensor \| None = None", "adj = None", _, flags=re.MULTILINE)
    _ = re.sub("adj_hat: torch\.Tensor \| None = None", "adj_hat = None", _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ###
    path = os.sep.join([base_dir, "src", "models", "base.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("embeddings: dict\[str, int \| None\]", "embeddings = None", _, flags=re.MULTILINE)
    _ = re.sub("node_ids: torch\.Tensor \| None = None", "node_ids = None", _, flags=re.MULTILINE)
    _ = re.sub("degrees: torch\.Tensor \| None = None", "degrees = None", _, flags=re.MULTILINE)
    _ = re.sub("embedding_dict: dict\[str, int \| None\]", "embedding_dict = None", _, flags=re.MULTILINE)
    _ = re.sub("degrees: np.ndarray \| None = None", "degrees = None", _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    ## MMR-GNN
    print("##### Integrating MMR-GNN #####")
    base_dir, base_import = get_bases("MMRGNN")
    path = os.sep.join([base_dir, "model.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^import util$", "from %s import util" % (base_import), _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)
    # Data
    ## Caltrans PeMS handler
    print("##### Integrating caltrans-pems #####")
    path = os.sep.join(["data", "caltrans_pems", "pems", "handler.py"])
    with open(path, "r", encoding="utf8") as f:
        _ = f.read()
    _ = re.sub("^from pems\.settings", "from data.caltrans_pems.pems.settings", _, flags=re.MULTILINE)
    with open(path, "w", encoding="utf8") as f:
        f.write(_)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Missing argument \"mode\" @ argv[1]. Options include [install_dependencies|integrate_submodules]")
    mode = sys.argv[1]
    if mode == "install_dependencies":
        kwargs = {}
        if len(sys.argv) > 2: kwargs["pytorch_version"] = sys.argv[2]
        if len(sys.argv) > 3: kwargs["cuda_version"] = sys.argv[3]
        if len(sys.argv) > 4: kwargs["debug"] = sys.argv[4]
        install_dependencies(**kwargs)
    elif mode == "integrate_submodules":
        kwargs = {}
        if len(sys.argv) > 2: kwargs["debug"] = sys.argv[2]
        integrate_submodules(**kwargs)
    else:
        raise ValueError(mode)
