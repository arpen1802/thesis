class XGBoostLightningModel(pl.LightningModule):
    def __init__(self, params_duration, params_energy):
        super().__init__()
        self.params_duration = params_duration
        self.params_energy = params_energy
        
        # Define the models
        self.model_duration = xgb.XGBRegressor(**self.params_duration)
        self.model_energy = xgb.XGBRegressor(**self.params_energy)

    def forward(self, x):
        # PyTorch Lightning expects a forward method
        # Predictions for duration and energy_need
        duration_preds = self.model_duration.predict(x)
        energy_preds = self.model_energy.predict(x)
        return duration_preds, energy_preds

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Split the targets
        y_duration = y[:, 0]
        y_energy = y[:, 1]
        
        # Train each model separately
        self.model_duration.fit(x, y_duration)
        self.model_energy.fit(x, y_energy)
        
        # Calculate predictions
        duration_preds, energy_preds = self.forward(x)
        
        # Calculate mean squared error loss
        duration_loss = mean_squared_error(y_duration, duration_preds)
        energy_loss = mean_squared_error(y_energy, energy_preds)
        loss = duration_loss + energy_loss
        
        # Log the losses
        self.log('train_duration_loss', duration_loss)
        self.log('train_energy_loss', energy_loss)
        self.log('train_total_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        # Split the targets
        y_duration = y[:, 0]
        y_energy = y[:, 1]
        
        # Calculate predictions
        duration_preds, energy_preds = self.forward(x)
        
        # Calculate mean squared error loss
        duration_loss = mean_squared_error(y_duration, duration_preds)
        energy_loss = mean_squared_error(y_energy, energy_preds)
        loss = duration_loss + energy_loss
        
        # Log the losses
        self.log('val_duration_loss', duration_loss)
        self.log('val_energy_loss', energy_loss)
        self.log('val_total_loss', loss)
        
        return loss

    def configure_optimizers(self):
        # XGBoost models handle optimization internally, no need for an optimizer here
        return None
