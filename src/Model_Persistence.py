import pickle


class ModelPersistence:

    @staticmethod
    def save(model, path):
        with open(path, "wb") as file:
            pickle.dump(model, file)

    @staticmethod
    def load(path):
        with open(path, "rb") as file:
            return pickle.load(file)