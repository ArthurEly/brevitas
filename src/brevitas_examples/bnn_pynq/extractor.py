import torch
import os

class ModelExtractor:
    def __init__(self, model, optimizer, checkpoints_dir_path, logger, args):
        self.model = model
        self.optimizer = optimizer
        self.checkpoints_dir_path = checkpoints_dir_path
        self.logger = logger
        self.args = args
        self.best_val_acc = 0.0

    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
        }, best_path)

    def load_model(self, checkpoint_path):
        """Carrega o modelo e o otimizador de um checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        epoch = checkpoint['epoch']
        return epoch

    def extract_and_save_model(self, checkpoint_path, save_path):
        """Extrai o modelo e salva em formato .pth."""
        self.load_model(checkpoint_path)
        # Salva apenas os pesos do modelo em formato .pth
        torch.save(self.model.state_dict(), save_path)
        self.logger.info("Model saved to {}".format(save_path))