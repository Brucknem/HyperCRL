import torch
import pytorch_lightning as pl
from torchvision import transforms

from hypercrl.srl.robotic_priors import RoboticPriors


class SRL(pl.LightningModule):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module = None):
        super().__init__()
        self.encoder = encoder
        self.has_decoder = decoder is not None
        if self.has_decoder:
            self.decoder = decoder
        self.robotic_priors = RoboticPriors()
        self.recon_loss_criterion = torch.nn.MSELoss()

    def forward(self, x):
        result = self.encoder(x)
        return result

    def log_loss(self, prefix: str, name: str, value: any):
        title = name
        if prefix != '':
            title = prefix + "/" + name
        title = "_" + title
        self.log(title, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def calculate_robotic_priors_loss(self, states, rewards):
        total_loss, slowness_loss, variability_loss, proportionality_loss, repeatability_loss, causality_loss = \
            self.robotic_priors(states[0]["z"], states[1]["z"], states[2]["z"], states[3]["z"],
                                rewards[0][1], rewards[1][1])

        self.log_loss("robotic_priors", "all", total_loss)
        self.log_loss("robotic_priors", "slowness_loss", slowness_loss)
        self.log_loss("robotic_priors", "variability_loss", variability_loss)
        self.log_loss("robotic_priors", "proportionality_loss", proportionality_loss)
        self.log_loss("robotic_priors", "repeatability_loss", repeatability_loss)
        self.log_loss("robotic_priors", "causality_loss", causality_loss)
        return total_loss

    @staticmethod
    def kl_loss(mu, log_var):
        return (-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)

    def calculate_kl_loss(self, states):
        for state in states:
            if "mu" not in state or "log_var" not in state:
                return torch.tensor(0)
        kl_losses = [self.kl_loss(state["mu"], state["log_var"]) for state in states]
        total_kl_loss = sum(kl_losses)
        self.log_loss("vae", "kl_loss", total_kl_loss)
        return total_kl_loss

    def calculate_reconstruction_loss(self, batch, states):
        reconstruction_losses = [
            self.recon_loss_criterion(batch['observations'][i], self.decoder(states[i]["z"])) for i in
            range(len(states))
        ]
        total_reconstruction_loss = sum(reconstruction_losses)
        self.log_loss("vae", "reconstruction_loss", total_reconstruction_loss)
        return total_reconstruction_loss

    def training_step(self, batch, batch_idx):
        states = [self.encoder(observation) for observation in batch['observations']]

        total_loss = 0
        total_loss += self.calculate_robotic_priors_loss(states, batch["rewards"])
        # total_loss += self.calculate_kl_loss(states)
        if self.has_decoder:
            total_loss += self.calculate_reconstruction_loss(batch, states)

        self.log_loss("", "_total_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
