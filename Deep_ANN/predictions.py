import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Switch the model to evaluation mode
model.eval()

# Disable gradient calculation for evaluation
with torch.no_grad():
    val_predictions = []
    val_targets = []
    for batch in val_loader:
        x, y = batch
        y_hat = model(x)
        val_predictions.append(y_hat)
        val_targets.append(y)

# Concatenate all the batches
val_predictions = torch.cat(val_predictions, dim=0)
val_targets = torch.cat(val_targets, dim=0)

# Convert to numpy arrays for easier manipulation
val_predictions = val_predictions.numpy()
val_targets = val_targets.numpy()

# Compute MSE and MAE
mse = mean_squared_error(val_targets, val_predictions)
mae = mean_absolute_error(val_targets, val_predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')

# Plot true vs. predicted values for time_delta
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(val_targets[:, 0], val_predictions[:, 0], alpha=0.5)
plt.xlabel('True time_delta')
plt.ylabel('Predicted time_delta')
plt.title('True vs. Predicted time_delta')

# Plot true vs. predicted values for kWh_delivered
plt.subplot(1, 2, 2)
plt.scatter(val_targets[:, 1], val_predictions[:, 1], alpha=0.5)
plt.xlabel('True kWh_delivered')
plt.ylabel('Predicted kWh_delivered')
plt.title('True vs. Predicted kWh_delivered')

plt.tight_layout()
plt.show()
