import jax
import jax.numpy as jnp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Forward function: Linear model
def model(params, X):
    W, b = params
    return jnp.dot(X, W) + b

# Training loop with early stopping and loss tracking
def train_one_fold(train_X, train_y, val_X, val_y, learning_rate=0.01, epochs=100, patience=10, noise_scale=0.0):
    # Initialize weights and biases randomly
    rng = jax.random.PRNGKey(42)
    W = jax.random.normal(rng, shape=(train_X.shape[1],))  # Weight vector
    b = jnp.zeros(1)  # Bias term

    # Parameter tuple
    params = (W, b)

    # Define the loss function (MSE in log-transformed space)
    def loss_fn(params, X, y):
        preds = model(params, X)
        return jnp.mean((preds - y) ** 2)  # MSE in transformed space

    # Compute gradients
    grad_fn = jax.grad(loss_fn)

    noise = jax.random.normal(rng, shape=params[0].shape) * noise_scale

    # Track losses
    train_losses = []
    val_losses = []

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop with tqdm progress bar
    progress_bar = tqdm(range(epochs), desc="Training Progress", unit="epoch")
    for epoch in progress_bar:
        grads = grad_fn(params, train_X, train_y)  # Gradients for W and b
        params = (
            params[0] - learning_rate * grads[0] + noise,  # Update W
            params[1] - learning_rate * grads[1],  # Update b
        )

        # Compute training and validation losses
        train_loss = loss_fn(params, train_X, train_y)
        val_loss = loss_fn(params, val_X, val_y)

        # Track losses
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))

        # Update tqdm progress bar with validation loss
        progress_bar.set_postfix(val_loss=f"{val_loss:.4f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best val_loss: {best_val_loss:.4f}")
            break

    return params, train_losses, val_losses

# Load your data (example dataset)
y = train["Premium Amount"].values
X = train.drop(columns=["Premium Amount", "outlier"], axis=1).values

# Transform the target variable
train["y_trans"] = np.log1p(y)

# Standardize features (optional, helps convergence for some models)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert to JAX arrays
X = jnp.array(X)
y_trans = jnp.array(train["y_trans"].values)

# Cross-validation setup
k = 3  # Number of splits
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Cross-validation loop
all_train_losses = []
all_val_losses = []

for fold_idx, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"\nFold {fold_idx + 1}/{k}")

    # Split into training and validation sets
    train_X, val_X = X[train_index], X[val_index]
    train_y, val_y = y_trans[train_index], y_trans[val_index]

    # Train the model for the current fold
    trained_params, train_losses, val_losses = train_one_fold(train_X, train_y, val_X, val_y)

    # Track losses for all folds
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)

    print(f"Finished Fold {fold_idx + 1}/{k}. Best Validation Loss: {min(val_losses):.4f}")

# Plot training vs validation losses for each fold
plt.figure(figsize=(12, 6))
for fold_idx in range(k):
    plt.plot(all_train_losses[fold_idx], label=f"Fold {fold_idx + 1} - Train Loss")
    plt.plot(all_val_losses[fold_idx], label=f"Fold {fold_idx + 1} - Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
