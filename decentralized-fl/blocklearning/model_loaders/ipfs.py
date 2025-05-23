from .diffusion.net import init_net
from .diffusion.diffusion_model import DiffusionModel, MODEL_NAME


class IpfsModelLoader:
    def __init__(
        self, contract, weights_loader, ipfs_api="/ip4/127.0.0.1/tcp/5001", partition=0
    ) -> None:
        self.contract = contract
        self.weights_loader = weights_loader
        self.ipfs_api = ipfs_api
        self.partition = partition
        pass

    def __load(self, weights_cid=""):
        print("__load")
        # with tempfile.TemporaryDirectory() as tempdir:
            # model_path = os.path.join(tempdir, "model.h5")
            # os.system(f"ipfs get --api {self.ipfs_api} -o {model_path} {model_cid}")

        model = init_net(MODEL_NAME)

        if weights_cid != "":
            weights = self.weights_loader.load(weights_cid)
            model.set_weights(weights)

        return DiffusionModel(model, self.partition)

    def load(self):
        # model_cid = self.contract.get_model()
        weights_cid = self.contract.get_weights(0)
        return self.__load(weights_cid)

    def load_top(self):
        # model_cid = self.contract.get_top_model()
        return self.__load()

    def load_bottom(self):
        # model_cid = self.contract.get_bottom_model()
        return self.__load()
