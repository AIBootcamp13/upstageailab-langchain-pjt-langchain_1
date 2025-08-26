class BaseDataset:
    def __init__(self, dataset_dir: str):
        self.docs = []
        self.dataset_dir = dataset_dir

    def load_docs(self):
        raise NotImplementedError()

    def get_docs(self):
        raise NotImplementedError()
