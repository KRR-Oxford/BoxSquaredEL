from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load_data(self, dataset):
        pass

    @abstractmethod
    def load_val_data(self, dataset, classes):
        pass

    @abstractmethod
    def load_test_data(self, dataset, classes):
        pass

    @staticmethod
    def from_task(task):
        from utils.old_data_loader import OldDataLoader
        from utils.inferences_data_loader import InferencesDataLoader
        from utils.prediction_data_loader import PredictionDataLoader
        task_to_data_loader = {
            'old': OldDataLoader,
            'inferences': InferencesDataLoader,
            'prediction': PredictionDataLoader
        }
        return task_to_data_loader[task]()
