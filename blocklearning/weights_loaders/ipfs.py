import os
import tempfile
import idx2numpy

MAX_WEIGHTS_PER_NODE = 2

class IpfsWeightsLoader:
    def __init__(self, ipfs_api="/ip4/127.0.0.1/tcp/5001"):
        self.ipfs_api = ipfs_api
        self.stored_weights = []

    def load(self, id):
        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = os.path.join(tempdir, "weights.idx")
            os.system(f"ipfs get --api {self.ipfs_api} -o {weights_path} {id}")
            weights = idx2numpy.convert_from_file(weights_path)
            return weights

    def store(self, weights):
        if self.stored_weights >= MAX_WEIGHTS_PER_NODE:
            os.system(f"ipfs --api {self.ipfs_api} pin rm {self.stored_weights.pop(0)}")
            os.system(f"ipfs --api {self.ipfs_api} repo gc")
        
        with tempfile.TemporaryDirectory() as tempdir:
            weights_path = os.path.join(tempdir, "weights.idx")
            idx2numpy.convert_to_file(weights_path, weights)
            weights_id = os.popen(f"ipfs add --api {self.ipfs_api} -q {weights_path}").read().strip().split("\n").pop()
            self.stored_weights.append(weights_id)
            return weights_id
